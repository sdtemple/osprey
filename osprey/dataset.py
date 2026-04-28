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
from .augment import (
    augmenter,
    p_augment,
    min_gain_db,
    max_gain_db,
)


class AudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        le, # LabelEncoder
        base_folder: str = base_folder,
        collection_map: dict[str, str] = collection_map,
        # mel spectrogram
        sr: int = sr,
        height_: int = height,
        width_: int = width,
        fmin_: float = fmin,
        fmax_: float = fmax,
        duration_seconds: float = duration,
        # Gain
        min_gain_db: float = min_gain_db,
        max_gain_db: float = max_gain_db,
        # probabilities of augmentation
        p_augment: float = p_augment,
        # p_color: float = 0.25,
        # p_timestretch: float = 0.25,
        # p_pitchshift: float = 0.25,
        # p_shift: float = 0.25,
        # p_gain: float = 0.25,
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
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.p_augment = p_augment

        # self.p_color = p_color
        # self.p_timestretch = p_timestretch
        # self.p_shift = p_shift
        # self.p_pitchshift = p_pitchshift
        # self.p_gain = p_gain

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
        # audio = augmenter(audio,
        #                   sr,
        #                   p_color=self.p_color,
        #                   p_gain=self.p_gain,
        #                   p_timestretch=self.p_timestretch,
        #                   p_shift=self.p_shift,
        #                   p_pitchshift=self.p_pitchshift,
        #                   min_gain_db=self.min_gain_db,
        #                   max_gain_db=self.max_gain_db,
        #                   )
        audio = augmenter(audio,
                          sr,
                          p_color=self.p_augment,
                          p_gain=self.p_augment,
                          p_timestretch=self.p_augment,
                          p_shift=self.p_augment,
                          p_pitchshift=self.p_augment,
                          min_gain_db=self.min_gain_db,
                          max_gain_db=self.max_gain_db,
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
