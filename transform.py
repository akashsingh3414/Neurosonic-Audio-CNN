import torch.nn as nn
import torchaudio.transforms as T
from augmentation import TimeShift, PitchShift, AddGaussianNoise, NormalizeSpec

class AudioTransforms:
    """Encapsulates training and validation transforms for ESC-50."""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

        self.train_transform = nn.Sequential(
            TimeShift(shift_max=0.2),
            PitchShift(sample_rate=sample_rate, n_steps=(-2, 2), p=0.7),
            AddGaussianNoise(noise_level=0.005),
            T.Resample(orig_freq=44100, new_freq=sample_rate),
            T.MelSpectrogram(
                sample_rate=sample_rate, 
                n_fft=2048, 
                hop_length=512, 
                n_mels=128,
                f_min=50,
                f_max=sample_rate // 2
            ),
            T.AmplitudeToDB(),
            NormalizeSpec(),
            T.FrequencyMasking(freq_mask_param=15),
            T.TimeMasking(time_mask_param=50),
        )

        self.val_transform = nn.Sequential(
            T.Resample(orig_freq=44100, new_freq=sample_rate),
            T.MelSpectrogram(
                sample_rate=sample_rate, 
                n_fft=2048, 
                hop_length=512, 
                n_mels=128,
                f_min=50,
                f_max=sample_rate // 2
            ),
            T.AmplitudeToDB(),
            NormalizeSpec(),
        )

    def get_transforms(self):
        """Return training and validation transforms"""
        return self.train_transform, self.val_transform
