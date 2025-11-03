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
class SpectrogramPitchShift(nn.Module):
    """
    Pitch shift by rolling mel bins (memory efficient).
    This approximates pitch shift on spectrograms.
    """
    def __init__(self, max_shift_bins=5, p=0.5):
        super().__init__()
        self.max_shift_bins = max_shift_bins
        self.p = p
    
    def forward(self, spectrogram):
        if not self.training or torch.rand(1).item() > self.p:
            return spectrogram
        
        shift = torch.randint(-self.max_shift_bins, self.max_shift_bins + 1, (1,)).item()
        if shift == 0:
            return spectrogram
        
        return torch.roll(spectrogram, shifts=shift, dims=1)

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