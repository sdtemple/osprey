from __future__ import annotations

import math
from typing import Any, Mapping

import librosa
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torchvision import transforms

base_folder = "/nfs/turbo/umor-sethtem/acoustics-data"
collection_map = {
    'iNat': 'birdclef-2026/train_audio',
    'XC': 'birdclef-2026/train_audio',
    'soundscape': 'birdclef-2026/train_soundscapes',
    'iNat2': 'birdclef-2025/train_audio',
    'XC2': 'birdclef-2025/train_audio',
    'CSA': 'birdclef-2025/train_audio',
    'esc': 'ESC-50-master/audio',
    'arca23k': 'ARCA23K/ARCA23K.audio/ARCA23K.audio',
    'urban8k': 'UrbanSound8K/audio/foldall',
    'dbr': 'dbr-dataset/dataset/dograin',
    'freesound': 'freesound/outside',
    'random-noise': 'random-noise',
    # "add": "additional_data",
}

def reformat_image(
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    """Prepare spectrogram data for pretrained image models.

    This keeps the spectrogram spatial dimensions unchanged and only expands
    grayscale inputs to RGB when needed before applying channel-wise normalization.
    """
    input_tensor = input_tensor.detach().clone().to(dtype=torch.float32)

    # add channel and/or batch dimension
    if len(input_tensor.shape) < 3:
        input_tensor = input_tensor.unsqueeze(0)
    if len(input_tensor.shape) < 4:
        input_tensor = input_tensor.unsqueeze(0)

    # repeat to rgb
    if input_tensor.shape[1] == 1:
        input_tensor = input_tensor.repeat(1, 3, 1, 1)
    elif input_tensor.shape[1] != 3:
        raise ValueError("Channel size is not 1 (grayscale) or 3 (RGB)")


    # normalize
    if input_tensor.max() > 1.0:
        input_tensor /= 255.0
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    return normalize(input_tensor)


def clean_row(row: pd.Series) -> dict[str, Any]:
    """Subset the useful info in each row."""
    row = row[
        [
            "primary_label",
            "start",
            "end",
            "filename",
            "collection",
            "latitude",
            "longitude",
            "dataset",
        ]
    ]
    return dict(row)


def get_audio(
    row: Mapping[str, Any],
    base_folder: str | None = None,
    collection_map: Mapping[str, str] | None = None,
    sr: int = 32000,
    duration: float = 5.,
) -> tuple[npt.NDArray[np.float32], int]:
    """Load audio for a metadata row."""
    if base_folder is None:
        base_folder = globals()["base_folder"]
    if collection_map is None:
        collection_map = globals()["collection_map"]
    assert base_folder is not None
    assert collection_map is not None

    fname = f"{base_folder}/{collection_map[row['collection']]}/{row['filename']}"
    y, sample_rate = librosa.load(
        fname,
        sr=sr,
        duration=duration,
        offset=row["start"],
    )
    return y, sample_rate


def get_mel(
    y: npt.NDArray,
    sr: int = 32000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 0,
    fmax: float = 16000,
) -> npt.NDArray:
    """Get a mel-spectrogram image."""
    x = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    x = librosa.power_to_db(x, ref=np.max)
    return x[-1::-1].copy()


def pad_mel_to_multiple(mel: npt.NDArray, multiple: int = 32) -> npt.NDArray:
    """Pad a 2D mel spectrogram to dimensions divisible by ``multiple``."""
    if mel.ndim != 2:
        raise ValueError("Expected a 2D mel spectrogram.")

    freq_pad = (-mel.shape[0]) % multiple
    time_pad = (-mel.shape[1]) % multiple

    top_pad = freq_pad
    bottom_pad = 0

    left_pad = np.random.randint(0, time_pad + 1) if time_pad > 0 else 0
    right_pad = time_pad - left_pad

    if freq_pad > 0 or time_pad > 0:
        mel = np.pad(
            mel,
            ((top_pad, bottom_pad), (left_pad, right_pad)),
            mode="constant",
            constant_values=0,
        )

    return mel


def pad_mel_spectrogram(
    mel: npt.NDArray,
    mel_time_size: int | None = None,
    multiple: int = 32,
) -> npt.NDArray:
    """Pad a mel spectrogram to a target time length and aligned size."""
    if mel_time_size is not None:
        current_time_size = mel.shape[-1]
        if current_time_size < mel_time_size:
            pad_amount = mel_time_size - current_time_size
            left_pad = np.random.randint(0, pad_amount + 1)
            right_pad = pad_amount - left_pad
            mel = np.pad(
                mel,
                ((0, 0), (left_pad, right_pad)),
                mode="constant",
                constant_values=0,
            )

    return pad_mel_to_multiple(mel, multiple)
