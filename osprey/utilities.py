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
fmin = 0
fmax = 16000
duration = 5
sr = 32000
n_fft = 2048
hop_length = 512
n_mels = 128

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

    # check gray or rgb
    if input_tensor.shape[1] in [1,3]:
        raise ValueError("Channel size is not 1 (grayscale) or 3 (RGB)")
    # repeat to rgb
    if input_tensor.shape[1] == 1:
        input_tensor = input_tensor.repeat(1, 3, 1, 1)

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
            "common_name",
            "sampling_rate_hz",
            "start_seconds",
            "end_seconds",
            "filename",
            "collection",
            "latitude",
            "longitude",
            "class_name",
            "dataset",
        ]
    ]
    return dict(row)


def get_audio(
    row: Mapping[str, Any],
    base_folder: str | None = None,
    collection_map: Mapping[str, str] | None = None,
    sr: int = sr,
    duration: float = duration,
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
        offset=row["start_seconds"],
    )
    return y, sample_rate


def get_mel(
    y: npt.NDArray,
    sr: int = sr,
    n_mels: int = n_mels,
    n_fft: int = n_fft,
    hop_length: int = hop_length,
    fmin: float = fmin,
    fmax: float = fmax,
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
