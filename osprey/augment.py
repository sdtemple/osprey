from __future__ import annotations

from random import randint
from audiomentations import Compose, AddColorNoise, TimeStretch, PitchShift, Shift, Gain
import numpy.typing as npt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as v2

p_augment = 0.25
min_gain_db = -12.
max_gain_db = 12.


### for spectrogram ###

class SpectrogramGain(nn.Module):
    """Custom transform to add logarithmic gain without overflowing uint8."""
    def __init__(self, min_gain=5, max_gain=25, max_value: float = 255.0):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.max_value = max_value
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generates a random scalar on the same GPU device as the tensor.
        # Use a float gain so notebook-level tuning can pass non-integer values safely.
        gain = torch.empty((), device=x.device, dtype=x.dtype).uniform_(
            float(self.min_gain),
            float(self.max_gain),
        )
        # Add gain and clamp to prevent overflow/saturation above 255
        return torch.clamp(x + gain, 0.0, float(self.max_value))
    
class SpectrogramShift(nn.Module):
    """
    A PyTorch module that applies a circular time shift to a batch of spectrograms.
    """
    def __init__(self, max_shift_pct: float = 0.15, dim: int = 3):
        """
        Parameters:
        -----------
        max_shift_pct : float
            The maximum percentage of the timeline the audio can shift.
            Defaults to 0.15 (15%).
        dim : int
            The dimension representing the time axis. 
            Defaults to 3 for standard 4D batches (Batch, Channel, Freq, Time).
            Set to 2 if applying to 3D single items (Channel, Freq, Time).
        """
        super().__init__()
        self.max_shift_pct = max_shift_pct
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Grab the total number of time steps
        time_steps = x.shape[self.dim]
        
        # 2. Calculate the maximum allowable shift in pixel columns
        max_shift = int(time_steps * self.max_shift_pct)
        
        # Guard against zero shift if spectrograms are extremely short
        if max_shift == 0:
            return x
            
        # 3. Generate a single random shift integer bounded by the percentage
        shift_amount = torch.randint(-max_shift, max_shift, (1,)).item()
        
        # 4. Perform the circular shift along the specified dimension
        return torch.roll(x, shifts=shift_amount, dims=self.dim)
    
class SpectrogramTimeMask(nn.Module):
    """
    A PyTorch module that applies a random time mask to a batch of spectrograms.
    """
    def __init__(self, 
                 max_mask_pct: float = 0.05, 
                 max_mask_num: int = 1,
                 dim: int = 3, 
                 fill_value: float = 0.0
                 ):
        """
        Parameters:
        -----------
        max_mask_pct : float
            The maximum percentage of the total timeline a single mask can cover.
        max_mask_num : int
            The maximum number of masks the total timeline a single mask can cover
        dim : int
            The dimension representing the time axis (last). 
        fill_value : float
            The value to fill the masked area with. 
        """
        super().__init__()
        self.max_mask_pct = max_mask_pct
        self.max_mask_num = max_mask_num
        self.dim = dim
        self.fill_value = fill_value
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Grab the total number of time steps
        time_steps = x.shape[self.dim]
        
        # 2. Calculate the maximum allowable mask size in pixel columns
        max_mask_len = int(time_steps * self.max_mask_pct)
        
        # Guard against zero mask length if spectrograms are extremely short
        if max_mask_len <= 1:
            return x
        
        # 3. Randomly determine how many masks to apply (up to max_mask_num)
        num_masks = torch.randint(1, self.max_mask_num + 1, (1,)).item()
        
        # 4. Clone once to avoid altering the grid in-place if tracking gradients
        x_masked = x.clone()
        
        # 5. Apply multiple masks
        for _ in range(num_masks):
            # Randomly determine the width of each mask
            mask_len = torch.randint(1, max_mask_len, (1,)).item()
            
            # Randomly determine where the mask starts (t0)
            t0 = torch.randint(0, time_steps - mask_len, (1,)).item()
            
            # Apply the mask across the entire batch
            if self.dim == 3:
                # For 4D batch: (Batch, Channel, Freq, Time)
                x_masked[:, :, :, t0 : t0 + mask_len] = self.fill_value
            elif self.dim == 2:
                # For 3D single item: (Channel, Freq, Time)
                x_masked[:, :, t0 : t0 + mask_len] = self.fill_value
            
        return x_masked
    
class SpectrogramFrequencyMask(nn.Module):
    """
    A PyTorch module that applies a random frequency mask to a batch of spectrograms.
    """
    def __init__(self, 
                 max_mask_len: int = 3, 
                 max_mask_num: int = 1,
                 dim: int = 2, 
                 fill_value: float = 0.0
                 ):
        """
        Parameters:
        -----------
        max_mask_len : int
            The maximum number of frequency bins a single mask can cover.
        max_mask_num : int
            The maximum number of masks the total timeline a single mask can cover
        dim : int
            The dimension representing the frequency axis (penultimate). 
        fill_value : float
            The value to fill the masked area with. 
        """
        super().__init__()
        self.max_mask_len = max_mask_len
        self.max_mask_num = max_mask_num
        self.dim = dim
        self.fill_value = fill_value
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Grab the total number of frequency steps
        freq_steps = x.shape[self.dim]
        
        # 2. Clamp the maximum mask size to the available frequency bins
        max_mask_len = min(int(self.max_mask_len), freq_steps)
        
        # Guard against zero mask length if spectrograms are extremely short
        if max_mask_len < 1:
            return x
        
        # 3. Randomly determine how many masks to apply (up to max_mask_num)
        num_masks = torch.randint(1, self.max_mask_num + 1, (1,)).item()
        
        # 4. Clone once to avoid altering the grid in-place if tracking gradients
        x_masked = x.clone()
        
        # 5. Apply multiple masks
        for _ in range(num_masks):
            # Randomly determine the width of each mask
            mask_len = torch.randint(1, max_mask_len + 1, (1,)).item()
            
            # Randomly determine where the mask starts (t0)
            t0 = torch.randint(0, freq_steps - mask_len, (1,)).item()
            
            # Apply the mask across the entire batch
            if self.dim == 2:
                # For 4D batch: (Batch, Channel, Freq, Time)
                x_masked[:, :, t0 : t0 + mask_len, :] = self.fill_value
            elif self.dim == 1:
                # For 3D single item: (Channel, Freq, Time)
                x_masked[:, t0 : t0 + mask_len, :] = self.fill_value
            
        return x_masked
    
def augmenter_spectrogram(x: torch.Tensor,
                          # SpectrogramGain
                          min_gain: float = 5.,
                          max_gain: float = 25.,
                          max_value: float = 255.,
                          p_gain: float = 0.25,
                          # SpectrogramShift
                          max_shift_pct: float = 0.05,
                          p_shift: float = 0.25,
                          # SpectrogramTimeMask
                          max_time_mask_pct: float = 0.02,
                          max_time_mask_num: int = 5,
                          p_time_mask: float = 0.25,
                          # SpectrogramFrequencyMask
                          max_freq_mask_len: int = 2,
                          max_freq_mask_num: int = 3,
                          p_freq_mask: float = 0.25,
                          ) -> torch.Tensor:
    """
    Augment a spectrogram tensor by composing multiple spectrogram transforms.
    
    Parameters:
    -----------
    x : torch.Tensor
        Input spectrogram tensor. Expected shape: (Batch, Channel, Freq, Time) or (Channel, Freq, Time)
    min_gain : float
        Minimum gain for SpectrogramGain
    max_gain : float
        Maximum gain for SpectrogramGain
    max_value : float
        Maximum allowed output value after gain is applied
    p_gain : float
        Probability of applying SpectrogramGain (0.0 to 1.0)
    max_shift_pct : float
        Maximum shift percentage for SpectrogramShift
    p_shift : float
        Probability of applying SpectrogramShift (0.0 to 1.0)
    max_time_mask_pct : float
        Maximum time mask percentage for SpectrogramTimeMask
    max_time_mask_num : int
        Maximum number of time masks
    p_time_mask : float
        Probability of applying SpectrogramTimeMask (0.0 to 1.0)
    max_freq_mask_len : int
        Maximum number of frequency bins a frequency mask can cover
    max_freq_mask_num : int
        Maximum number of frequency masks
    p_freq_mask : float
        Probability of applying SpectrogramFrequencyMask (0.0 to 1.0)
    
    Returns:
    --------
    torch.Tensor
        Augmented spectrogram tensor with the same shape as input
    """
    is_batched = x.dim() == 4
    time_dim = 3 if is_batched else 2
    freq_dim = 2 if is_batched else 1
    
    transforms = v2.Compose([
        v2.RandomApply([SpectrogramGain(min_gain=min_gain, max_gain=max_gain, max_value=max_value)], 
                       p=p_gain,
                       ),
        # v2.RandomApply([SpectrogramShift(max_shift_pct=max_shift_pct, dim=time_dim)], 
        #                p=p_shift,
        #                ),
        v2.RandomApply([SpectrogramTimeMask(max_mask_pct=max_time_mask_pct, 
                                            max_mask_num=max_time_mask_num, 
                                            dim=time_dim)], 
                       p=p_time_mask,
                       ),
        v2.RandomApply([SpectrogramFrequencyMask(max_mask_len=max_freq_mask_len, 
                                                 max_mask_num=max_freq_mask_num, 
                                                 dim=freq_dim)], 
                       p=p_freq_mask,
                       ),
    ])
    
    return transforms(x)
    


### for waveform ###

def augmenter_waveform(y: npt.NDArray,
                        sr: int = 32000,
                        # AddColorNoise
                        min_snr_db: float = 5.,
                        max_snr_db: float = 5.,
                        p_color: float = 0.33,
                        # TimeStretch
                        min_rate: float = 0.8,
                        max_rate: float = 1.25,
                        p_timestretch: float = 0.33,
                        # PitchShift
                        min_semitones: float = -4.,
                        max_semitones: float = 4.,
                        p_pitchshift: float = 0.33,
                        # Shift
                        min_shift: float = -0.5,
                        max_shift: float = 0.5,
                        p_shift: float = 0.33,
                        # Gain
                        min_gain_db: float = -12.,
                        max_gain_db: float = 12.,
                        p_gain: float = 0.33,
                        ):
    """Augment waveform according to audiomentations package"""
    
    def in_between(_):
        return _ >= 0. and _ <= 1.
    assert in_between(p_color)
    assert in_between(p_timestretch)
    assert in_between(p_pitchshift)
    assert in_between(p_shift)
    assert in_between(p_gain)
    
    colors = [
        _ for _ in range(-6,7,1)
    ]
    n_colors = len(colors)
    color = colors[randint(0, n_colors - 1)]

    augment = Compose([

        AddColorNoise(min_snr_db=min_snr_db,
                      max_snr_db=max_snr_db,
                      p=p_color,
                      ),
    
        # Stretch the duration of the sound slightly (speed up or slow down) with 20% probability
        # rate 0.8 (slower) to 1.25 (faster)
        TimeStretch(min_rate=min_rate, 
                    max_rate=max_rate, 
                    p=p_timestretch,
                    ),
    
        # Shift the pitch slightly up or down by a few semi-tones with 60% probability
        # -4 semi-tones to +4 semi-tones
        PitchShift(min_semitones=min_semitones, 
                   max_semitones=max_semitones, 
                   p=p_pitchshift,
                   ),
    
        # Shift the time axis of the sound slightly with 70% probability
        # shift_ms can be positive or negative (forwards or backwards in time within the clip)
        Shift(min_shift=min_shift, 
              max_shift=max_shift, 
              p=p_shift,
              ),
        
        # Adjust the volume (gain) up or down slightly with 40% probability
        # min_gain_in_db=-6 (quieter) to max_gain_in_db=6 (louder)
        Gain(min_gain_db=min_gain_db, 
             max_gain_db=max_gain_db, 
             p=p_gain,
             ), 
        
    ])

    return augment(samples=y, sample_rate=sr)