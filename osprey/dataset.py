from __future__ import annotations

import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

from .utilities import (
    clean_row,
    duration,
    fmax,
    fmin,
    get_audio,
    get_mel,
    height,
    base_folder,
    collection_map,
    sr,
    width,
)


class AudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        le,
        base_folder: str = base_folder,
        collection_map: dict[str, str] = collection_map,
        sr: int = sr,
        height_: int = height,
        width_: int = width,
        fmin_: float = fmin,
        fmax_: float = fmax,
        duration_seconds: float = duration,
    ) -> None:
        """
        Create an audio dataset backed by a dataframe.

        Parameters
        ----------
            df : pd.DataFrame
            le : sklearn.LabelEncoder
        """
        self.df = df.reset_index(drop=True)
        self.le = le
        self.base_folder = base_folder
        self.collection_map = collection_map
        self.sr = sr
        self.height = height_
        self.width = width_
        self.fmin = fmin_
        self.fmax = fmax_
        self.duration = duration_seconds

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):

        # Get the row data
        row = self.df.iloc[idx]
        row = clean_row(row)

        # Process the audio
        audio, sr = get_audio(
            row,
            base_folder=self.base_folder,
            collection_map=self.collection_map,
            sr=self.sr,
            duration_seconds=self.duration,
        )
        x, _ = get_mel(
            audio,
            sr,
            height_=self.height,
            width_=self.width,
            fmin_=self.fmin,
            fmax_=self.fmax,
            duration_seconds=self.duration,
        )
        x = librosa.power_to_db(x, ref=np.max)
        
        # Convert to tensor
        x_tensor = torch.from_numpy(x).float()
        x_tensor = x_tensor.unsqueeze(0)
        y = row['primary_label']
        y = self.le.transform([y])[0]
        y = torch.tensor(y)
        # One-hot logic here

        return x_tensor, y
