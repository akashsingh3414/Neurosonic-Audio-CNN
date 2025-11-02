import torch
import torch.nn as nn
import torchaudio
import numpy as np
import random

class NormalizeSpec(nn.Module):
    """Normalize spectrograms"""
    def forward(self, spec):
        mean = spec.mean()
        std = spec.std()
        return (spec - mean) / (std + 1e-6)

class TimeShift(nn.Module):
    """Time shifting augmentation"""
    def __init__(self, shift_max=0.2):
        super().__init__()
        self.shift_max = shift_max
    
    def forward(self, waveform):
        if self.training:
            shift = int(np.random.uniform(-self.shift_max, self.shift_max) * waveform.shape[-1])
            return torch.roll(waveform, shift, dims=-1)
        return waveform

class PitchShift(torch.nn.Module):
    """
    Randomly pitch-shift an audio waveform.
    Args:
        sample_rate (int): Audio sampling rate (e.g., 44100 or 22050)
        n_steps (tuple or list): Range of pitch shift in semitones, e.g. (-2, 2)
        p (float): Probability of applying the shift
    """
    def __init__(self, sample_rate=44100, n_steps=(-2, 2), p=0.5):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_steps = n_steps
        self.p = p

    def forward(self, waveform):
        # waveform: (channels, time)
        if random.random() > self.p:
            return waveform  # no change

        # Pick random shift in semitones
        steps = random.uniform(self.n_steps[0], self.n_steps[1])

        # Apply torchaudio pitch shift
        shifted = torchaudio.functional.pitch_shift(
            waveform, 
            sample_rate=self.sample_rate, 
            n_steps=steps
        )
        return shifted

class AddGaussianNoise(nn.Module):
    """Add Gaussian noise to waveform"""
    def __init__(self, noise_level=0.005):
        super().__init__()
        self.noise_level = noise_level
    
    def forward(self, waveform):
        if self.training and np.random.random() > 0.5:
            noise = torch.randn_like(waveform) * self.noise_level
            return waveform + noise
        return waveform
