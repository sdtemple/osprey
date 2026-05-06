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
    augmenter_waveform,
    augmenter_spectrogram,
    p_augment,
    min_gain_db,
    max_gain_db,
)

### gpu bound ###


class SpectrogramDatasetGPU(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        le, # LabelEncoder
        base_folder: str = base_folder,
        collection_map: dict[str, str] = collection_map,
    ) -> None:
        """
        Create a spectrogram dataset from precomputed .npz files.

        Parameters
        ----------
            df : pd.DataFrame
                Dataframe with audio metadata. Must have a 'filename' column or similar
                that can be used to construct the path to the .npz file.
            le : sklearn.LabelEncoder
                Label encoder for class labels.
            npz_folder : str
                Path to folder containing .npz files.
        """
        self.df = df.reset_index(drop=True)
        self.le = le
        self.base_folder = base_folder
        self.collection_map = collection_map

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # Get the row data
        row = self.df.iloc[idx]
        row = clean_row(row)
        
        # Construct path to .npz file (assumes filename column exists)
        npz_file = f"{self.base_folder}/{self.collection_map[row['collection']]}/{row['filename']}"
        
        # Load spectrogram from .npz file
        x = np.load(npz_file)['spectrogram']
        
        # Convert to tensor
        x_tensor = torch.from_numpy(x).float()
        x_tensor = x_tensor.unsqueeze(0)  # Add channel dimension
        
        # Get label
        y = row['primary_label']
        y = self.le.transform([y])[0]
        y = torch.tensor(y)
        
        return x_tensor, y


### cpu bound ###


class AudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        le, # LabelEncoder
        base_folder: str = base_folder,
        collection_map: dict[str, str] = collection_map,
        # mel spectrogram
        sr: int = sr,
        height: int = height,
        width: int = width,
        fmin: float = fmin,
        fmax: float = fmax,
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
        self.height = height
        self.width = width
        self.fmin = fmin
        self.fmax = fmax
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
            duration=self.duration,
        )
        audio = augmenter_waveform(audio,
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
            height=self.height,
            width=self.width,
            fmin=self.fmin,
            fmax=self.fmax,
            duration=self.duration,
        )
        # Convert to tensor
        x_tensor = torch.from_numpy(x).float()
        x_tensor = x_tensor.unsqueeze(0)
        y = row['primary_label']
        y = self.le.transform([y])[0]
        y = torch.tensor(y)
        # One-hot logic here

        return x_tensor, y


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        le, # LabelEncoder
        base_folder: str = base_folder,
        collection_map: dict[str, str] = collection_map,
        # Spectrogram augmentation
        p_augment: float = p_augment,
        min_gain: float = 5.,
        max_gain: float = 25.,
        p_gain: float = 0.25,
        max_shift_pct: float = 0.20,
        p_shift: float = 0.25,
        max_time_mask_pct: float = 0.02,
        max_time_mask_num: int = 5,
        p_time_mask: float = 0.25,
        max_freq_mask_len: int = 3,
        max_freq_mask_num: int = 5,
        p_freq_mask: float = 0.25,
    ) -> None:
        """
        Create a spectrogram dataset from precomputed .npz files.

        Parameters
        ----------
            df : pd.DataFrame
                Dataframe with audio metadata. Must have a 'filename' column or similar
                that can be used to construct the path to the .npz file.
            le : sklearn.LabelEncoder
                Label encoder for class labels.
            npz_folder : str
                Path to folder containing .npz files.
            p_augment : float
                Base probability for augmentation (used if individual p_* not specified).
            min_gain : float
                Minimum gain for SpectrogramGain.
            max_gain : float
                Maximum gain for SpectrogramGain.
            p_gain : float
                Probability of applying SpectrogramGain.
            max_shift_pct : float
                Maximum shift percentage for SpectrogramShift.
            p_shift : float
                Probability of applying SpectrogramShift.
            max_time_mask_pct : float
                Maximum time mask percentage for SpectrogramTimeMask.
            max_time_mask_num : int
                Maximum number of time masks.
            p_time_mask : float
                Probability of applying SpectrogramTimeMask.
            max_freq_mask_len : int
                Maximum number of frequency bins a frequency mask can cover.
            max_freq_mask_num : int
                Maximum number of frequency masks.
            p_freq_mask : float
                Probability of applying SpectrogramFrequencyMask.
        """
        self.df = df.reset_index(drop=True)
        self.le = le
        self.p_augment = p_augment
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.p_gain = p_gain
        self.max_shift_pct = max_shift_pct
        self.p_shift = p_shift
        self.max_time_mask_pct = max_time_mask_pct
        self.max_time_mask_num = max_time_mask_num
        self.p_time_mask = p_time_mask
        self.max_freq_mask_len = max_freq_mask_len
        self.max_freq_mask_num = max_freq_mask_num
        self.p_freq_mask = p_freq_mask
        self.base_folder = base_folder
        self.collection_map = collection_map

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # Get the row data
        row = self.df.iloc[idx]
        row = clean_row(row)
        
        # Construct path to .npz file (assumes filename column exists)
        npz_file = f"{self.base_folder}/{self.collection_map[row['collection']]}/{row['filename']}"
        
        # Load spectrogram from .npz file
        x = np.load(npz_file)['spectrogram']
        
        # Convert to tensor
        x_tensor = torch.from_numpy(x).float()
        x_tensor = x_tensor.unsqueeze(0)  # Add channel dimension
        
        # Apply augmentation
        x_tensor = augmenter_spectrogram(
            x_tensor,
            p_gain=self.p_augment,
            p_shift=self.p_augment,
            p_time_mask=self.p_augment,
            p_freq_mask=self.p_augment,
        )
        # x_tensor = augmenter_spectrogram(
        #     x_tensor,
        #     min_gain=self.min_gain,
        #     max_gain=self.max_gain,
        #     p_gain=self.p_gain,
        #     max_shift_pct=self.max_shift_pct,
        #     p_shift=self.p_shift,
        #     max_time_mask_pct=self.max_time_mask_pct,
        #     max_time_mask_num=self.max_time_mask_num,
        #     p_time_mask=self.p_time_mask,
        #     max_freq_mask_pct=self.max_freq_mask_pct,
        #     max_freq_mask_num=self.max_freq_mask_num,
        #     p_freq_mask=self.p_freq_mask,
        # )
        
        # Get label
        y = row['primary_label']
        y = self.le.transform([y])[0]
        y = torch.tensor(y)
        
        return x_tensor, y
