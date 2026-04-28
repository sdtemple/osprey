from __future__ import annotations

from random import randint
from audiomentations import Compose, AddColorNoise, TimeStretch, PitchShift, Shift, Gain
import numpy.typing as npt
import numpy as np

p_augment = 0.25
min_gain_db = -12.
max_gain_db = 12.

def augmenter(y: npt.NDArray,
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