from __future__ import annotations

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from random import sample, randint

from .utilities import (
    clean_row,
    get_audio,
    get_mel,
    base_folder,
    collection_map,
    pad_mel_spectrogram,
)
from .augment import (
    augmenter_waveform,
)

### gpu bound ###


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        le, # LabelEncoder
        base_folder: str = base_folder,
        collection_map: dict[str, str] = collection_map,
        mel_time_size: int | None = None,
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
            mel_time_size : int | None
                Target time dimension size for mel spectrograms. If specified, spectrograms
                shorter than this will be randomly padded. If None, no padding is applied.
        """
        self.df = df.reset_index(drop=True)
        self.le = le
        self.base_folder = base_folder
        self.collection_map = collection_map
        self.mel_time_size = mel_time_size
        self.num_classes = len(le.classes_)

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
        
        x = pad_mel_spectrogram(x, mel_time_size=self.mel_time_size)
        
        # Convert to tensor
        x_tensor = torch.from_numpy(x).float()
        x_tensor = x_tensor.unsqueeze(0)  # Add channel dimension
        
        # Get label
        y = row['primary_label'].split(';')
        y_idx = self.le.transform(y)
        y_tensor = F.one_hot(torch.tensor(y_idx, dtype=torch.long), num_classes=self.num_classes).float()
        
        return x_tensor, y_tensor
    
class SpectrogramOverlayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        le, # LabelEncoder
        base_folder: str = base_folder,
        collection_map: dict[str, str] = collection_map,
        mel_time_size: int | None = None,
        max_number: int = 4,
    ) -> None:
        """
        Create a spectrogram dataset with overlayed spectrogram samples.

        Parameters
        ----------
            df : pd.DataFrame
                Dataframe with audio metadata, including mixed label and filename columns.
            le : sklearn.LabelEncoder
                Label encoder for class labels.
            base_folder : str
                Root folder used to locate spectrogram files.
            collection_map : dict[str, str]
                Mapping from collection identifiers to folder names.
            mel_time_size : int | None
                Target time dimension size for mel spectrograms.
            max_number : int
                Maximum number of overlay samples to draw for each item.
        """
        self.df = df.reset_index(drop=True)
        self.le = le
        self.base_folder = base_folder
        self.collection_map = collection_map
        self.mel_time_size = mel_time_size
        self.num_classes = len(le.classes_)
        self.max_number = max_number

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # Get the row data
        row = self.df.iloc[idx]
        mix_primary_labels = np.array(row.mix_primary_labels.split(';'))
        mix_filenames = np.array(row.mix_filenames.split(';'))
        mix_collections = np.array(row.mix_collections.split(';'))
        n_overlay = len(mix_filenames)
        row = clean_row(row)
        
        # sample the overlay files
        overlay_indices = sample(range(n_overlay), k=self.max_number)
        mix_filenames = mix_filenames[overlay_indices]
        mix_primary_labels = mix_primary_labels[overlay_indices]
        mix_collections = mix_collections[overlay_indices]

        # Construct path to .npz file (assumes filename column exists)
        npz_file = f"{self.base_folder}/{self.collection_map[row['collection']]}/{row['filename']}"
        npz_overlay_files = [
          f"{self.base_folder}/{self.collection_map[j]}/{i}"  for i, j in zip(mix_filenames, mix_collections)
        ]
        
        # Load spectrogram from .npz file
        x = np.load(npz_file)['spectrogram']
        x = pad_mel_spectrogram(x, mel_time_size=self.mel_time_size)
        u = np.array(
            [
                pad_mel_spectrogram(np.load(_)['spectrogram'], self.mel_time_size) for _ in npz_overlay_files 
            ]
        )
        
        # Convert to tensor
        x_tensor = torch.from_numpy(x).float()
        x_tensor = x_tensor.unsqueeze(0)  # Add channel dimension
        u_tensor = torch.from_numpy(u).float()
        u_tensor = u_tensor.unsqueeze(0)
        
        # Get label
        y = row['primary_label'].split(';')
        y_idx = self.le.transform(y)
        v_idx = self.le.transform(mix_primary_labels)

        y_tensor = F.one_hot(
            torch.tensor(y_idx, dtype=torch.long),
            num_classes=self.num_classes,
        ).float().sum(dim=0).clamp(max=1.0)
        v_tensor = F.one_hot(
            torch.tensor(v_idx, dtype=torch.long),
            num_classes=self.num_classes,
        ).float() # don't combine the other spectrograms yet
        
        return x_tensor, y_tensor, u_tensor, v_tensor 


### cpu bound ###


class AudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        le, # LabelEncoder
        base_folder: str = base_folder,
        collection_map: dict[str, str] = collection_map,
        sr: int = 32000,
        duration: float = 5.,
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
        self.num_classes = len(le.classes_)

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
        y = row['primary_label'].split(';')
        y_idx = self.le.transform(y)
        y_tensor = F.one_hot(torch.tensor(y_idx, dtype=torch.long), num_classes=self.num_classes).float()

        return x_tensor, y_tensor


def waveform_batch_to_mel(
    waveforms: torch.Tensor,
    sr: int = 32000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 0,
    fmax: float = 16000,
    duration: float = 5.,
    apply_waveform_augment: bool = False,
    p_augment: float = 0.25,
    min_gain_db: float = -12,
    max_gain_db: float = 12,
    mel_time_size: int | None = None,
) -> torch.Tensor:
    """Convert a batch of waveforms into mel spectrogram tensors.
    
    Parameters
    ----------
        waveforms : torch.Tensor
            Batch of waveforms.
        mel_time_size : int | None
            Target time dimension size for mel spectrograms. If specified, spectrograms
            shorter than this will be randomly padded. If None, no padding is applied.
    """
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
        )

        mel = pad_mel_spectrogram(mel, mel_time_size=mel_time_size)
        
        mel_batch.append(torch.from_numpy(mel).float().unsqueeze(0))

    return torch.stack(mel_batch, dim=0)
