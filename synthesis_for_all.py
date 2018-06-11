# coding: utf-8
"""
Synthesis waveform from trained model.

usage: synthesis.py [options]

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --preset=<json>                   Path of preset parameters (json).
    --data-root=<dir>            Directory contains preprocessed features.
    --text_list_file=<name>            Text list file.
    --dst_dir=<dir>            Directory to output voices.
    --checkpoint_dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    -h, --help               Show help message.
"""
from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext

import audio

import torch
from torch.autograd import Variable
import numpy as np
import nltk
import random
import librosa
from nnmnkwii.datasets import vctk

# The deepvoice3 model
from deepvoice3_pytorch import frontend
from hparams import hparams, hparams_debug_string
from train import load_checkpoint, build_speech_semi_autoencoder, PyTorchDataset, \
    FileSourceDataset, PartialyRandomizedSimilarTimeLengthSampler, collate_fn, \
    LinearSpecDataSource, MelSpecDataSource, TextDataSource
from shutil import copyfile

from torch.utils import data as data_utils
from torch import optim

from tqdm import tqdm

assert torch.cuda.is_available()
use_cuda = True
_frontend = None  # to be set later


def tts(model, voice_input, text, p=0, fast=False):
    """Convert text to speech waveform given a deepvoice3 model.

    Args:
        text (str) : Input text to be synthesized
        p (float) : Replace word to pronounciation if p > 0. Default is 0.
    """
    model = model.cuda()
    model.eval()
    if fast:
        model.make_generation_fast_()

    voice = Variable(torch.from_numpy(voice_input)).transpose(0, 1).unsqueeze(0).unsqueeze(0)
    sequence = np.array(_frontend.text_to_sequence(text, p=p))
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0).long()
    text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long()
    text_positions = Variable(text_positions)
    voice = voice.cuda()
    sequence = sequence.cuda()
    text_positions = text_positions.cuda()

    # Greedy decoding
    mel_outputs, linear_outputs, alignments, done = model(
        sequence, voice_input=voice, text_positions=text_positions)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio._denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()
    mel = mel_outputs[0].cpu().data.numpy()
    mel = audio._denormalize(mel)

    # Predicted audio signal
    waveform = audio.inv_spectrogram(linear_output.T)

    return waveform, alignment, spectrogram, mel


def _load(checkpoint_path):
    return torch.load(checkpoint_path)


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint_dir"]
    text_list_file_path = args["--text_list_file"]
    dst_dir = args["--dst_dir"]
    speaker_id = None
    preset = args["--preset"]
    data_root = args["--data-root"]

    replace_pronunciation_prob = 0.0
    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "deepvoice3"

    _frontend = getattr(frontend, hparams.frontend)
    import train

    train._frontend = _frontend
    from train import plot_alignment

    reset_optimizer = False
    X = FileSourceDataset(TextDataSource(data_root, speaker_id))
    Mel = FileSourceDataset(MelSpecDataSource(data_root, speaker_id))
    Y = FileSourceDataset(LinearSpecDataSource(data_root, speaker_id))

    os.makedirs(dst_dir, exist_ok=True)  # parent of all directory

    # this checkpoint path points to the path that contains all old checkpoints
    # we're going to take each of them, load it into model and synthesize some voices
    for checkpoint in os.listdir(checkpoint_dir):
        if not checkpoint.startswith('checkpoint') or checkpoint[-9] not in ('0', '5'):
            continue

        checkpoint_load_path = os.path.abspath(join(checkpoint_dir, checkpoint))

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

        load_checkpoint(checkpoint_load_path, model, optimizer, encoder_optimizer, reset_optimizer)

        current_model_dir = join(dst_dir, checkpoint)
        os.makedirs(current_model_dir, exist_ok=True)

        raw_to_processed_map = {speaker: idx for idx, speaker in enumerate(vctk.available_speakers)}
        processed_to_raw_map = {idx: speaker for idx, speaker in enumerate(vctk.available_speakers)}

        for i in range(5):
            random_voice_idx = random.randint(0, len(Mel))
            random_speaker_id = X[random_voice_idx][1]
            lin = Y[random_voice_idx]
            mel = Mel[random_voice_idx]

            speaker_dir = join(current_model_dir, 'speaker{}'.format(random_speaker_id))
            os.makedirs(speaker_dir, exist_ok=True)

            audio.save_wav(audio.inv_spectrogram(lin.T), join(speaker_dir, 'sample_voice.wav'))

            with open(text_list_file_path, "rb") as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    text = line.decode("utf-8")[:-1]
                    waveform, alignment, _, _ = tts(
                        model, mel, text, p=replace_pronunciation_prob, fast=False)
                    dst_wav_path = join(speaker_dir, "text{}.wav".format(idx))
                    audio.save_wav(waveform, dst_wav_path)

        print("Finished! Check out {} for generated audio samples.".format(dst_dir))

    sys.exit(0)
