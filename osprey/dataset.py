from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from .utilities import (
    clean_row,
    duration,
    fmax,
    fmin,
    n_fft,
    n_mels,
    hop_length,
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
    p_augment,
    min_gain_db,
    max_gain_db,
)

### gpu bound ###


class SpectrogramDataset(Dataset):
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
        sr: int = sr,
        duration: float = duration,
    ) -> None:
        """
        Create an audio dataset backed by a dataframe.
        
        Returns raw waveforms; augmentation and mel conversion handled by waveform_batch_to_mel.

        Parameters
        ----------
            df : pd.DataFrame
                Dataframe with audio metadata.
            le : sklearn.LabelEncoder
                Label encoder for class labels.
            base_folder : str
                Root path for audio files.
            collection_map : dict[str, str]
                Mapping from collection name to relative path.
            sr : int
                Sampling rate for audio loading.
            duration : float
                Duration in seconds for audio clips.
        """
        self.df = df.reset_index(drop=True)
        self.le = le
        self.base_folder = base_folder
        self.collection_map = collection_map
        self.sr = sr
        self.duration = duration

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):

        # Get the row data
        row = self.df.iloc[idx]
        row = clean_row(row)

        # Process the audio
        audio, _ = get_audio(
            row,
            base_folder=self.base_folder,
            collection_map=self.collection_map,
            sr=self.sr,
            duration=self.duration,
        )

        # Return raw waveform so augmentation and mel conversion can run in-loop.
        target_num_samples = int(self.sr * self.duration)
        if len(audio) < target_num_samples:
            audio = np.pad(audio, (0, target_num_samples - len(audio)))
        elif len(audio) > target_num_samples:
            audio = audio[:target_num_samples]

        x_tensor = torch.from_numpy(audio).float()
        y = row['primary_label']
        y = self.le.transform([y])[0]
        y = torch.tensor(y)

        return x_tensor, y


def waveform_batch_to_mel(
    waveforms: torch.Tensor,
    sr: int = sr,
    n_mels: int = n_mels,
    n_fft: int = n_fft,
    hop_length: int = hop_length,
    fmin: float = fmin,
    fmax: float = fmax,
    duration: float = duration,
    apply_waveform_augment: bool = False,
    p_augment: float = p_augment,
    min_gain_db: float = min_gain_db,
    max_gain_db: float = max_gain_db,
) -> torch.Tensor:
    """Convert a batch of waveforms into mel spectrogram tensors."""
    if waveforms.dim() == 1:
        waveforms = waveforms.unsqueeze(0)
    elif waveforms.dim() == 3 and waveforms.shape[1] == 1:
        waveforms = waveforms.squeeze(1)
    elif waveforms.dim() != 2:
        raise ValueError("Expected waveform tensor with shape [B, T], [B, 1, T], or [T].")

    mel_batch: list[torch.Tensor] = []
    waveforms_np = waveforms.detach().cpu().numpy().astype(np.float32, copy=False)

    for waveform in waveforms_np:
        if apply_waveform_augment and p_augment > 0.0:
            waveform = augmenter_waveform(
                waveform,
                sr=sr,
                p_color=p_augment,
                p_gain=p_augment,
                p_timestretch=p_augment,
                p_shift=p_augment,
                p_pitchshift=p_augment,
                min_gain_db=min_gain_db,
                max_gain_db=max_gain_db,
            )

        mel = get_mel(
            waveform,
            sr=sr,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            duration=duration,
        )
        mel_batch.append(torch.from_numpy(mel).float().unsqueeze(0))

    return torch.stack(mel_batch, dim=0)
