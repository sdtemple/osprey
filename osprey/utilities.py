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
    "iNat": "birdclef-2026/train_audio",
    "XC": "birdclef-2026/train_audio",
    "esc": "ESC-50-master/ESC-50-master/audio",
    "arca23k": "ARCA23K/ARCA23K.audio/ARCA23K.audio",
    "urban8k": "UrbanSound8K/audio/foldall",
    "dbr": "dbr-dataset/dataset/dograin",
    "freesound": "freesound/outside",
    "random_noise": "random-noise",
    # "add": "additional_data",
}

height = 224
width = 224
fmin = 0
fmax = 16000
duration = 5
sr = 32000
n_fft = 2048
hop_length = 512
n_mels = 128
image_size = height, width


def reformat_image(
    input_tensor: torch.Tensor,
    image_size: tuple[int, int] = image_size,
    channel_size: int = 3,
) -> torch.Tensor:
    """Prepare spectrogram data for pretrained image models."""
    input_tensor = input_tensor.detach().clone().to(dtype=torch.float32)

    if len(input_tensor.shape) < 3:
        input_tensor = input_tensor.unsqueeze(0)
    if len(input_tensor.shape) < 4:
        input_tensor = input_tensor.unsqueeze(0)

    if input_tensor.shape[1] != channel_size:
        input_tensor = input_tensor.repeat(1, channel_size, 1, 1)

    if input_tensor.max() > 1.0:
        input_tensor /= 255.0

    if channel_size == 3:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    elif channel_size == 1:
        normalize = transforms.Normalize(mean=[0.449], std=[0.226])
    else:
        raise ValueError("Channel size is not 1 (grayscale) or 3 (RGB)")

    if input_tensor.shape[-2:] != image_size:
        preproc = transforms.Compose(
            [
                transforms.Resize(
                    image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                normalize,
            ]
        )
    else:
        preproc = transforms.Compose([normalize])

    return preproc(input_tensor)


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
        base_folder = base_folder
    if collection_map is None:
        collection_map = collection_map

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
    duration: float = duration,
) -> tuple[npt.NDArray, int]:
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
    mn = np.min(x)
    return x[-1::-1].copy()
