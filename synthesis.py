# coding: utf-8
"""
Synthesis waveform from trained model.

usage: synthesis.py [options] <checkpoint> <text_list_file> <dst_dir> <wav_path>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --preset=<json>                   Path of preset parameters (json).
    -h, --help               Show help message.
"""
import librosa
from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext

import audio

import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import nltk

# The deepvoice3 model
from deepvoice3_pytorch import frontend
from hparams import hparams, hparams_debug_string

from tqdm import tqdm

assert torch.cuda.is_available()
use_cuda = True
_frontend = None  # to be set later


def _process_utterance(wav_path):
    sr = hparams.sample_rate

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    wav, _ = librosa.effects.trim(wav, top_db=15)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    # n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    # Return a tuple describing this training example:
    return spectrogram.T, mel_spectrogram.T


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
    checkpoint_path = args["<checkpoint>"]
    text_list_file_path = args["<text_list_file>"]
    dst_dir = args["<dst_dir>"]
    wav_path = args["<wav_path>"]
    speaker_id = None
    preset = args["--preset"]

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

    # Model
    model = train.build_speech_semi_autoencoder()
    model = model.cuda()

    encoder_optimizer = optim.Adam(
        model.encoder.parameters(), lr=hparams.initial_learning_rate,
        betas=(hparams.adam_beta1, hparams.adam_beta2), eps=hparams.adam_eps,
        weight_decay=hparams.weight_decay)

    optimizer = optim.Adam(
        model.get_trainable_parameters(), lr=hparams.initial_learning_rate,
        betas=(hparams.adam_beta1, hparams.adam_beta2), eps=hparams.adam_eps,
        weight_decay=hparams.weight_decay)

    train.load_checkpoint(checkpoint_path, model, optimizer, encoder_optimizer,
                          reset_optimizer=False)

    checkpoint_name = splitext(basename(checkpoint_path))[0]

    mel = _process_utterance(wav_path=wav_path)[1]
    os.makedirs(dst_dir, exist_ok=True)
    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            text = line.decode("utf-8")[:-1]
            words = nltk.word_tokenize(text)
            waveform, alignment, _, _ = tts(
                model, mel, text, p=0.0, fast=False)
            dst_wav_path = join(dst_dir, "{}_{}.wav".format(
                idx, checkpoint_name))
            audio.save_wav(waveform, dst_wav_path)
            from os.path import basename, splitext

            name = splitext(basename(text_list_file_path))[0]

    sys.exit(0)
