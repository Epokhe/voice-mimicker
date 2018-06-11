"""Trainining script for seq2seq text-to-speech synthesis model.

usage: train.py [options]

options:
    --data-root=<dir>            Directory contains preprocessed features.
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --checkpoint-seq2seq=<path>  Restore seq2seq model from checkpoint path.
    --checkpoint-postnet=<path>  Restore postnet model from checkpoint path.
    --train-seq2seq-only         Train only seq2seq model.
    --train-postnet-only         Train only postnet model.
    --restore-parts=<path>       Restore part of the model.
    --log-event-path=<name>      Log event path.
    --reset-optimizer            Reset optimizer.
    --load-embedding=<path>      Load embedding from checkpoint.
    --speaker-id=<N>             Use specific speaker of data in case for multi-speaker datasets.
    -h, --help                   Show this help message and exit
"""
import matplotlib

matplotlib.use('Agg')
import json

from docopt import docopt

import sys
import gc
import platform
from os.path import dirname, join
from tqdm import tqdm, trange
from datetime import datetime

# The deepvoice3 model
from deepvoice3_pytorch import frontend
import audio
import lrschedule

import torch
from torch.utils import data as data_utils
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler
import numpy as np
from numba import jit

from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from os.path import join, expanduser
import random

import librosa.display
from matplotlib import pyplot as plt
import sys
import os
from tensorboardX import SummaryWriter
from matplotlib import cm
from warnings import warn
from hparams import hparams, hparams_debug_string
from model import DeepSpeech, SpeechSemiAutoEncoder, deepvoice3_multispeaker, SPEAKER_EMBED_DIM

fs = hparams.sample_rate

global_step = 0
global_epoch = 0
assert torch.cuda.is_available()
use_cuda = True
cudnn.benchmark = False

_frontend = None  # to be set later

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


def plot_alignment(alignment, path, info=None):
    fig, ax = plt.subplots()
    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()


class TextDataSource(FileDataSource):
    def __init__(self, data_root, speaker_id=None):
        self.data_root = data_root
        self.speaker_ids = None
        self.multi_speaker = False
        # If not None, filter by speaker_id
        self.speaker_id = speaker_id

    def collect_files(self):
        meta = join(self.data_root, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 4 or len(l) == 5
        self.multi_speaker = len(l) == 5
        texts = list(map(lambda l: l.decode("utf-8").split("|")[3], lines))
        if self.multi_speaker:
            speaker_ids = list(map(lambda l: int(l.decode("utf-8").split("|")[-1]), lines))
            # Filter by speaker_id
            # using multi-speaker dataset as a single speaker dataset
            if self.speaker_id is not None:
                indices = np.array(speaker_ids) == self.speaker_id
                texts = list(np.array(texts)[indices])
                self.multi_speaker = False
                return texts

            return texts, speaker_ids
        else:
            return texts

    def collect_features(self, *args):
        if self.multi_speaker:
            text, speaker_id = args
        else:
            text = args[0]
        global _frontend
        if _frontend is None:
            _frontend = getattr(frontend, hparams.frontend)
        seq = _frontend.text_to_sequence(text, p=hparams.replace_pronunciation_prob)

        if platform.system() == "Windows":
            if hasattr(hparams, 'gc_probability'):
                _frontend = None  # memory leaking prevention in Windows
                if np.random.rand() < hparams.gc_probability:
                    gc.collect()  # garbage collection enforced
                    print("GC done")

        if self.multi_speaker:
            return np.asarray(seq, dtype=np.int32), int(speaker_id)
        else:
            return np.asarray(seq, dtype=np.int32)


class _NPYDataSource(FileDataSource):
    def __init__(self, data_root, col, speaker_id=None):
        self.data_root = data_root
        self.col = col
        self.frame_lengths = []
        self.speaker_id = speaker_id

    def collect_files(self):
        meta = join(self.data_root, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 4 or len(l) == 5
        multi_speaker = len(l) == 5
        self.frame_lengths = list(
            map(lambda l: int(l.decode("utf-8").split("|")[2]), lines))

        paths = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.data_root, f), paths))

        if multi_speaker and self.speaker_id is not None:
            speaker_ids = list(map(lambda l: int(l.decode("utf-8").split("|")[-1]), lines))
            # Filter by speaker_id
            # using multi-speaker dataset as a single speaker dataset
            indices = np.array(speaker_ids) == self.speaker_id
            paths = list(np.array(paths)[indices])
            self.frame_lengths = list(np.array(self.frame_lengths)[indices])
            # aha, need to cast numpy.int64 to int
            self.frame_lengths = list(map(int, self.frame_lengths))

        self.frame_lengths = self.frame_lengths[:9 * len(self.frame_lengths) // 10]

        return paths[:9 * len(paths) // 10]

    def collect_features(self, path):
        return np.load(path)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, speaker_id=None):
        super(MelSpecDataSource, self).__init__(data_root, 1, speaker_id)


class LinearSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, speaker_id=None):
        super(LinearSpecDataSource, self).__init__(data_root, 0, speaker_id)


class PartialyRandomizedSimilarTimeLengthSampler(Sampler):
    """Partially randmoized sampler

    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batchs
    """

    def __init__(self, lengths, batch_size=16, batch_group_size=None,
                 permutate=True):
        self.lengths, self.sorted_indices = torch.sort(torch.LongTensor(lengths))
        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 32, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0
        self.permutate = permutate

    def __iter__(self):
        indices = self.sorted_indices.clone()
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            random.shuffle(indices[s:e])

        # Permutate batches
        if self.permutate:
            perm = np.arange(len(indices[:e]) // self.batch_size)
            random.shuffle(perm)
            indices[:e] = indices[:e].view(-1, self.batch_size)[perm, :].view(-1)

        # Handle last elements
        s += batch_group_size
        if s < len(indices):
            random.shuffle(indices[s:])

        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)


class PyTorchDataset(object):
    def __init__(self, X, Mel, Y):
        self.X = X
        self.Mel = Mel
        self.Y = Y
        self.speaker_indices = {}
        for idx, (_, speaker_id) in enumerate(X):
            try:
                self.speaker_indices[speaker_id].append(idx)
            except KeyError:
                self.speaker_indices[speaker_id] = [idx]
        # alias
        self.multi_speaker = X.file_data_source.multi_speaker

    # note that due to second voice of the speaker, output is random, same index is not guaranteed
    # to be the same between two epochs
    def __getitem__(self, idx):
        text, speaker_id = self.X[idx]
        same_speaker_random_Mel = self.Mel[random.choice(self.speaker_indices[speaker_id])]

        return text, self.Mel[idx], self.Y[idx], speaker_id, same_speaker_random_Mel

    def __len__(self):
        return len(self.X)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss(size_average=False)

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, D)
        mask_ = mask.expand_as(input)
        loss = self.criterion(input * mask_, target * mask_)
        return loss / mask_.sum()


def collate_fn(batch):
    """Create batch"""

    r = hparams.outputs_per_step
    downsample_step = hparams.downsample_step

    # Lengths
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)

    # use this in STT net
    voice_input_lengths = [len(x[4]) for x in batch]

    target_lengths = [len(x[1]) for x in batch]

    max_target_len = max(target_lengths)
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0
    if max_target_len % downsample_step != 0:
        max_target_len += downsample_step - max_target_len % downsample_step
        assert max_target_len % downsample_step == 0

    max_voice_input_len = max(voice_input_lengths)
    if max_voice_input_len % r != 0:
        max_voice_input_len += r - max_voice_input_len % r
        assert max_voice_input_len % r == 0
    if max_voice_input_len % downsample_step != 0:
        max_voice_input_len += downsample_step - max_voice_input_len % downsample_step
        assert max_voice_input_len % downsample_step == 0

    # Set 0 for zero beginning padding
    # imitates initial decoder states
    b_pad = r
    max_target_len += b_pad * downsample_step

    max_voice_input_len += b_pad * downsample_step

    a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)
    target_lengths = torch.LongTensor(target_lengths)
    voice_input_lengths = torch.LongTensor(voice_input_lengths)

    b = np.array([_pad_2d(x[1], max_target_len, b_pad=b_pad) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[2], max_target_len, b_pad=b_pad) for x in batch],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)

    d = np.array([_pad_2d(x[4], max_voice_input_len, b_pad=b_pad) for x in batch],
                 dtype=np.float32)
    voice_input_batch = torch.FloatTensor(d)
    voice_input_batch = voice_input_batch.transpose(1, 2).unsqueeze(1)

    # text positions
    text_positions = np.array([_pad(np.arange(1, len(x[0]) + 1), max_input_len)
                               for x in batch], dtype=np.int)
    text_positions = torch.LongTensor(text_positions)

    max_decoder_target_len = max_target_len // r // downsample_step

    # frame positions
    s, e = 1, max_decoder_target_len + 1
    # if b_pad > 0:
    #    s, e = s - 1, e - 1
    frame_positions = torch.arange(s, e).long().unsqueeze(0).expand(
        len(batch), max_decoder_target_len)

    # done flags
    done = np.array([_pad(np.zeros(len(x[1]) // r // downsample_step - 1),
                          max_decoder_target_len, constant_values=1)
                     for x in batch])
    done = torch.FloatTensor(done).unsqueeze(-1)

    speaker_ids = torch.LongTensor([x[3] for x in batch])

    return x_batch, input_lengths, mel_batch, y_batch, \
           (text_positions, frame_positions), done, target_lengths, speaker_ids, voice_input_batch


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def save_alignment(path, attn):
    plot_alignment(attn.T, path, info="{}, {}, step={}".format(
        hparams.builder, time_string(), global_step))


def prepare_spec_image(spectrogram):
    # [0, 1]
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    spectrogram = np.flip(spectrogram, axis=1)  # flip against freq axis
    return np.uint8(cm.magma(spectrogram.T) * 255)


def eval_model(global_step, writer, model, checkpoint_dir, ismultispeaker, x_dataset, mel_dataset):
    for i in range(5):
        texts = [
            # "Scientists at the CERN laboratory say they have discovered a new particle.",
            # "There's a way to measure the acute emotional intelligence that has never gone out of style.",
            # "President Trump met with other leaders at the Group of 20 conference.",
            # "Generative adversarial network or variational auto-encoder.",
            "Please call Stella.",
            "Some have accepted this as a miracle without any physical explanation.",
        ]
        random_voice_idx = random.randint(0, len(mel_dataset))
        speaker_id = x_dataset[random_voice_idx][1]
        voice = mel_dataset[random_voice_idx]

        import synthesis
        synthesis._frontend = _frontend

        eval_output_dir = join(checkpoint_dir, "eval")
        os.makedirs(eval_output_dir, exist_ok=True)

        # hard coded
        # speaker_ids = [0, 1, 10] if ismultispeaker else [None]
        # for speaker_id in speaker_ids:
        speaker_str = "multispeaker{}".format(speaker_id) if speaker_id is not None else "single"

        for idx, text in enumerate(texts):
            signal, alignment, _, mel = synthesis.tts(model, voice, text, p=0, fast=False)
            signal /= np.max(np.abs(signal))

            # Alignment
            path = join(eval_output_dir, "step{:09d}_text{}_{}_alignment.png".format(
                global_step, idx, speaker_str))
            save_alignment(path, alignment)
            tag = "eval_averaged_alignment_{}_{}".format(idx, speaker_str)
            writer.add_image(tag, np.uint8(cm.viridis(np.flip(alignment, 1).T) * 255), global_step)

            # Mel
            writer.add_image("(Eval) Predicted mel spectrogram text{}_{}".format(idx, speaker_str),
                             prepare_spec_image(mel), global_step)

            # Audio
            path = join(eval_output_dir, "step{:09d}_text{}_{}_predicted.wav".format(
                global_step, idx, speaker_str))
            audio.save_wav(signal, path)

            try:
                pass
                writer.add_audio("(Eval) Predicted audio signal {}_{}".format(idx, speaker_str),
                                 signal, global_step, sample_rate=fs)
            except Exception as e:
                warn(str(e))
                pass


def save_states(global_step, writer, mel_outputs, linear_outputs, attn, mel, y,
                input_lengths, checkpoint_dir=None):
    print("Save intermediate states at step {}".format(global_step))

    # idx = np.random.randint(0, len(input_lengths))
    idx = min(1, len(input_lengths) - 1)
    input_length = input_lengths[idx]

    # Alignment
    # Multi-hop attention
    if attn is not None and attn.dim() == 4:
        for i, alignment in enumerate(attn):
            alignment = alignment[idx].cpu().data.numpy()
            tag = "alignment_layer{}".format(i + 1)
            writer.add_image(tag, np.uint8(cm.viridis(np.flip(alignment, 1).T) * 255), global_step)

            # save files as well for now
            alignment_dir = join(checkpoint_dir, "alignment_layer{}".format(i + 1))
            os.makedirs(alignment_dir, exist_ok=True)
            path = join(alignment_dir, "step{:09d}_layer_{}_alignment.png".format(
                global_step, i + 1))
            save_alignment(path, alignment)

        # Save averaged alignment
        alignment_dir = join(checkpoint_dir, "alignment_ave")
        os.makedirs(alignment_dir, exist_ok=True)
        path = join(alignment_dir, "step{:09d}_alignment.png".format(global_step))
        alignment = attn.mean(0)[idx].cpu().data.numpy()
        save_alignment(path, alignment)

        tag = "averaged_alignment"
        writer.add_image(tag, np.uint8(cm.viridis(np.flip(alignment, 1).T) * 255), global_step)

    # Predicted mel spectrogram
    if mel_outputs is not None:
        mel_output = mel_outputs[idx].cpu().data.numpy()
        mel_output = prepare_spec_image(audio._denormalize(mel_output))
        writer.add_image("Predicted mel spectrogram", mel_output, global_step)

    # Predicted spectrogram
    if linear_outputs is not None:
        linear_output = linear_outputs[idx].cpu().data.numpy()
        spectrogram = prepare_spec_image(audio._denormalize(linear_output))
        writer.add_image("Predicted linear spectrogram", spectrogram, global_step)

        # Predicted audio signal
        signal = audio.inv_spectrogram(linear_output.T)
        signal /= np.max(np.abs(signal))
        path = join(checkpoint_dir, "step{:09d}_predicted.wav".format(
            global_step))
        try:
            writer.add_audio("Predicted audio signal", signal, global_step, sample_rate=fs)
        except Exception as e:
            warn(str(e))
            pass
        audio.save_wav(signal, path)

    # Target mel spectrogram
    if mel_outputs is not None:
        mel_output = mel[idx].cpu().data.numpy()
        mel_output = prepare_spec_image(audio._denormalize(mel_output))
        writer.add_image("Target mel spectrogram", mel_output, global_step)

    # Target spectrogram
    if linear_outputs is not None:
        linear_output = y[idx].cpu().data.numpy()
        spectrogram = prepare_spec_image(audio._denormalize(linear_output))
        writer.add_image("Target linear spectrogram", spectrogram, global_step)


def logit(x, eps=1e-8):
    return torch.log(x + eps) - torch.log(1 - x + eps)


def masked_mean(y, mask):
    # (B, T, D)
    mask_ = mask.expand_as(y)
    return (y * mask_).sum() / mask_.sum()


def spec_loss(y_hat, y, mask, priority_bin=None, priority_w=0):
    masked_l1 = MaskedL1Loss()
    l1 = nn.L1Loss()

    w = hparams.masked_loss_weight

    # L1 loss
    if w > 0:
        assert mask is not None
        l1_loss = w * masked_l1(y_hat, y, mask=mask) + (1 - w) * l1(y_hat, y)
    else:
        assert mask is None
        l1_loss = l1(y_hat, y)

    # Priority L1 loss
    if priority_bin is not None and priority_w > 0:
        if w > 0:
            priority_loss = w * masked_l1(
                y_hat[:, :, :priority_bin], y[:, :, :priority_bin], mask=mask) \
                            + (1 - w) * l1(y_hat[:, :, :priority_bin], y[:, :, :priority_bin])
        else:
            priority_loss = l1(y_hat[:, :, :priority_bin], y[:, :, :priority_bin])
        l1_loss = (1 - priority_w) * l1_loss + priority_w * priority_loss

    # Binary divergence loss
    if hparams.binary_divergence_weight <= 0:
        binary_div = Variable(y.data.new(1).zero_())
    else:
        y_hat_logits = logit(y_hat)
        z = -y * y_hat_logits + torch.log(1 + torch.exp(y_hat_logits))
        if w > 0:
            binary_div = w * masked_mean(z, mask) + (1 - w) * z.mean()
        else:
            binary_div = z.mean()

    return l1_loss, binary_div


@jit(nopython=True)
def guided_attention(N, max_N, T, max_T, g):
    W = np.zeros((max_N, max_T), dtype=np.float32)
    for n in range(N):
        for t in range(T):
            W[n, t] = 1 - np.exp(-(n / N - t / T) ** 2 / (2 * g * g))
    return W


def guided_attentions(input_lengths, target_lengths, max_target_len, g=0.2):
    B = len(input_lengths)
    max_input_len = input_lengths.max()
    W = np.zeros((B, max_target_len, max_input_len), dtype=np.float32)
    for b in range(B):
        W[b] = guided_attention(input_lengths[b], max_input_len,
                                target_lengths[b], max_target_len, g).T
    return W


def train(model, data_loader, optimizer, encoder_optimizer, writer,
          init_lr=0.002, checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0, x_dataset=None, mel_dataset=None):
    model = model.cuda()
    linear_dim = model.linear_dim
    r = hparams.outputs_per_step
    downsample_step = hparams.downsample_step
    current_lr = init_lr

    binary_criterion = nn.BCELoss()

    global global_step, global_epoch
    while global_epoch < nepochs:
        running_loss = 0.
        for step, (x, input_lengths, mel, y, positions, done, target_lengths,
                   speaker_ids, voice) \
                in tqdm(enumerate(data_loader)):
            model.train()
            is_multispeaker = speaker_ids is not None
            # Learning rate schedule
            if hparams.lr_schedule is not None:
                lr_schedule_f = getattr(lrschedule, hparams.lr_schedule)
                current_lr = lr_schedule_f(
                    init_lr, global_step, **hparams.lr_schedule_kwargs)
                if hparams.encoder_optimize_ratio:
                    for param_group in encoder_optimizer.param_groups:
                        param_group['lr'] = current_lr

                if not hparams.encoder_optimize_ratio or (
                        global_step > 0 and global_step % hparams.encoder_optimize_ratio == 0):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr

            if hparams.encoder_optimize_ratio:
                encoder_optimizer.zero_grad()
            if not hparams.encoder_optimize_ratio or (
                    global_step > 0 and global_step % hparams.encoder_optimize_ratio == 0):
                optimizer.zero_grad()

            # Used for Position encoding
            text_positions, frame_positions = positions

            # Downsample mel spectrogram
            if downsample_step > 1:
                mel = mel[:, 0::downsample_step, :].contiguous()

            # Lengths
            input_lengths = input_lengths.long().numpy()
            decoder_lengths = target_lengths.long().numpy() // r // downsample_step

            max_seq_len = max(input_lengths.max(), decoder_lengths.max())
            if max_seq_len >= hparams.max_positions:
                raise RuntimeError(
                    """max_seq_len ({}) >= max_posision ({})
                    Input text or decoder targget length exceeded the maximum length.
                    Please set a larger value for ``max_position`` in hyper parameters.""".format(
                        max_seq_len, hparams.max_positions))

            # Feed data
            voice, x, mel, y = Variable(voice), Variable(x), Variable(mel), Variable(y)
            text_positions = Variable(text_positions)
            frame_positions = Variable(frame_positions)
            done = Variable(done)
            target_lengths = Variable(target_lengths)

            text_positions = text_positions.cuda()
            frame_positions = frame_positions.cuda()
            voice, x, mel, y = voice.cuda(), x.cuda(), mel.cuda(), y.cuda()
            done, target_lengths = done.cuda(), target_lengths.cuda()

            # Create mask if we use masked loss
            if hparams.masked_loss_weight > 0:
                # decoder output domain mask
                decoder_target_mask = sequence_mask(
                    target_lengths / (r * downsample_step),
                    max_len=mel.size(1)).unsqueeze(-1)
                if downsample_step > 1:
                    # spectrogram-domain mask
                    target_mask = sequence_mask(
                        target_lengths, max_len=y.size(1)).unsqueeze(-1)
                else:
                    target_mask = decoder_target_mask
                # shift mask
                decoder_target_mask = decoder_target_mask[:, r:, :]
                target_mask = target_mask[:, r:, :]
            else:
                decoder_target_mask, target_mask = None, None

            # Apply model
            mel_outputs, linear_outputs, attn, done_hat = model(
                x, mel,
                text_positions=text_positions, frame_positions=frame_positions,
                input_lengths=input_lengths, voice_input=voice)

            # Losses
            w = hparams.binary_divergence_weight

            # mel:
            mel_l1_loss, mel_binary_div = spec_loss(
                mel_outputs[:, :-r, :], mel[:, r:, :], decoder_target_mask)
            mel_loss = (1 - w) * mel_l1_loss + w * mel_binary_div

            # done:
            done_loss = binary_criterion(done_hat, done)

            # linear:
            n_priority_freq = int(hparams.priority_freq / (fs * 0.5) * linear_dim)
            linear_l1_loss, linear_binary_div = spec_loss(
                linear_outputs[:, :-r, :], y[:, r:, :], target_mask,
                priority_bin=n_priority_freq,
                priority_w=hparams.priority_freq_weight)
            linear_loss = (1 - w) * linear_l1_loss + w * linear_binary_div

            # Combine losses
            loss = mel_loss + linear_loss + done_loss

            # attention
            if hparams.use_guided_attention:
                soft_mask = guided_attentions(input_lengths, decoder_lengths,
                                              attn.size(-2),
                                              g=hparams.guided_attention_sigma)
                soft_mask = Variable(torch.from_numpy(soft_mask))
                soft_mask = soft_mask.cuda()
                attn_loss = (attn * soft_mask).mean()
                loss += attn_loss

            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_states(
                    global_step, writer, mel_outputs, linear_outputs, attn,
                    mel, y, input_lengths, checkpoint_dir)
                save_checkpoint(model, optimizer, encoder_optimizer, global_step, checkpoint_dir,
                                global_epoch)

            if global_step > 0 and global_step % hparams.eval_interval == 0:
                eval_model(global_step, writer, model, checkpoint_dir, is_multispeaker, x_dataset,
                           mel_dataset)

            # Update
            loss.backward()
            if clip_thresh > 0:
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.get_trainable_parameters(), clip_thresh)

            if hparams.encoder_optimize_ratio:
                encoder_optimizer.step()
            if not hparams.encoder_optimize_ratio or (
                    global_step > 0 and global_step % hparams.encoder_optimize_ratio == 0):
                optimizer.step()

            # Logs
            writer.add_scalar("loss", float(loss.data[0]), global_step)
            writer.add_scalar("done_loss", float(done_loss.data[0]), global_step)
            writer.add_scalar("mel loss", float(mel_loss.data[0]), global_step)
            writer.add_scalar("mel_l1_loss", float(mel_l1_loss.data[0]), global_step)
            writer.add_scalar("mel_binary_div_loss", float(mel_binary_div.data[0]), global_step)
            writer.add_scalar("linear_loss", float(linear_loss.data[0]), global_step)
            writer.add_scalar("linear_l1_loss", float(linear_l1_loss.data[0]), global_step)
            writer.add_scalar("linear_binary_div_loss", float(
                linear_binary_div.data[0]), global_step)
            if hparams.use_guided_attention:
                writer.add_scalar("attn_loss", float(attn_loss.data[0]), global_step)
            if clip_thresh > 0:
                writer.add_scalar("gradient norm", grad_norm, global_step)
            writer.add_scalar("learning rate", current_lr, global_step)

            global_step += 1
            running_loss += loss.data[0]

        averaged_loss = running_loss / (len(data_loader))
        writer.add_scalar("loss (per epoch)", averaged_loss, global_epoch)
        print("Loss: {}".format(running_loss / (len(data_loader))))

        global_epoch += 1


def save_checkpoint(model, optimizer, encoder_optimizer, step, checkpoint_dir, epoch):
    suffix = ""
    m = model

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}{}.pth".format(global_step, suffix))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    encoder_optimizer_state = encoder_optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": m.state_dict(),
        "optimizer": optimizer_state,
        "encoder_optimizer": encoder_optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def build_speech_semi_autoencoder():
    return SpeechSemiAutoEncoder(encoder=build_deepspeech_model(),
                                 decoder=build_deepvoice_model())


def build_deepvoice_model():
    model = deepvoice3_multispeaker(
        n_speakers=hparams.n_speakers,
        speaker_embed_dim=SPEAKER_EMBED_DIM,
        n_vocab=_frontend.n_vocab,
        embed_dim=hparams.text_embed_dim,
        mel_dim=hparams.num_mels,
        linear_dim=hparams.fft_size // 2 + 1,
        r=hparams.outputs_per_step,
        downsample_step=hparams.downsample_step,
        padding_idx=hparams.padding_idx,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        encoder_channels=hparams.encoder_channels,
        decoder_channels=hparams.decoder_channels,
        converter_channels=hparams.converter_channels,
        use_memory_mask=hparams.use_memory_mask,
        trainable_positional_encodings=hparams.trainable_positional_encodings,
        force_monotonic_attention=hparams.force_monotonic_attention,
        use_decoder_state_for_postnet_input=hparams.use_decoder_state_for_postnet_input,
        max_positions=hparams.max_positions,
        speaker_embedding_weight_std=hparams.speaker_embedding_weight_std,
        freeze_embedding=hparams.freeze_embedding,
        window_ahead=hparams.window_ahead,
        window_backward=hparams.window_backward,
        key_projection=hparams.key_projection,
        value_projection=hparams.value_projection,
    )
    return model


def build_deepspeech_model():
    sample_rate = 16000
    window_size = .02
    window_stride = .01
    window = 'hamming'
    noise_dir = None
    noise_prob = 0.4
    noise_min = 0.0
    noise_max = 0.5
    audio_conf = dict(sample_rate=sample_rate,
                      window_size=window_size,
                      window_stride=window_stride,
                      window=window,
                      noise_dir=noise_dir,
                      noise_prob=noise_prob,
                      noise_levels=(noise_min, noise_max))

    hidden_size = 100
    hidden_layers = 5
    labels_path = 'labels.json'

    rnn_type = 'gru'
    bidirectional = True
    model = DeepSpeech(rnn_hidden_size=hidden_size,
                       nb_layers=hidden_layers,
                       rnn_type=supported_rnns[rnn_type],
                       audio_conf=audio_conf,
                       bidirectional=bidirectional)
    return model


def _load(checkpoint_path):
    return torch.load(checkpoint_path)


def load_checkpoint(path, model, optimizer, encoder_optimizer, reset_optimizer):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
        encoder_optimizer_state = checkpoint["encoder_optimizer"]
        if encoder_optimizer_state is not None:
            print("Load encoder optimizer state from {}".format(path))
            encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model


def _load_embedding(path, model):
    state = _load(path)["state_dict"]
    key = "seq2seq.encoder.embed_tokens.weight"
    model.seq2seq.encoder.embed_tokens.weight.data = state[key]


# https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3


def restore_parts(path, model):
    print("Restore part of the model from: {}".format(path))
    state = _load(path)["state_dict"]
    model_dict = model.state_dict()
    valid_state_dict = {k: v for k, v in state.items() if k in model_dict}

    try:
        model_dict.update(valid_state_dict)
        model.load_state_dict(model_dict)
    except RuntimeError as e:
        # there should be invalid size of weight(s), so load them per parameter
        print(str(e))
        model_dict = model.state_dict()
        for k, v in valid_state_dict.items():
            model_dict[k] = v
            try:
                model.load_state_dict(model_dict)
            except RuntimeError as e:
                print(str(e))
                warn("{}: may contain invalid size of weight. skipping...".format(k))


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    speaker_id = None
    preset = args["--preset"]

    data_root = args["--data-root"]

    reset_optimizer = False

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])

    assert hparams.name == "deepvoice3"
    print(hparams_debug_string())

    _frontend = getattr(frontend, hparams.frontend)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Input dataset definitions
    X = FileSourceDataset(TextDataSource(data_root, speaker_id))
    Mel = FileSourceDataset(MelSpecDataSource(data_root, speaker_id))
    Y = FileSourceDataset(LinearSpecDataSource(data_root, speaker_id))

    # Prepare sampler
    frame_lengths = Mel.file_data_source.frame_lengths
    sampler = PartialyRandomizedSimilarTimeLengthSampler(
        frame_lengths, batch_size=hparams.batch_size)

    # Dataset and Dataloader setup
    dataset = PyTorchDataset(X, Mel, Y)
    data_loader = data_utils.DataLoader(
        dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, sampler=sampler,
        collate_fn=collate_fn, pin_memory=hparams.pin_memory)

    # Model
    model = build_speech_semi_autoencoder()
    model = model.cuda()

    encoder_optimizer = optim.Adam(
        model.encoder.parameters(), lr=hparams.initial_learning_rate,
        betas=(hparams.adam_beta1, hparams.adam_beta2), eps=hparams.adam_eps,
        weight_decay=hparams.weight_decay)

    optimizer = optim.Adam(
        model.get_trainable_parameters(), lr=hparams.initial_learning_rate,
        betas=(hparams.adam_beta1, hparams.adam_beta2), eps=hparams.adam_eps,
        weight_decay=hparams.weight_decay)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, encoder_optimizer, reset_optimizer)

    # Setup summary writer for tensorboard
    log_event_path = "log/run-test" + str(datetime.now()).replace(" ", "_")
    print("Los event path: {}".format(log_event_path))
    writer = SummaryWriter(log_dir=log_event_path)

    # Train!
    train(model, data_loader, optimizer, encoder_optimizer, writer,
          init_lr=hparams.initial_learning_rate,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.checkpoint_interval,
          nepochs=hparams.nepochs,
          clip_thresh=hparams.clip_thresh,
          x_dataset=X,
          mel_dataset=Mel)

    print("Finished")
    sys.exit(0)
