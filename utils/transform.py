import torch.nn as nn
import torchaudio.transforms as T
from utils.augmentation import TimeShift, SpectrogramPitchShift, AddGaussianNoise, NormalizeSpec

class AudioTransforms:
    """Encapsulates training and validation transforms for ESC-50."""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

        self.train_transform = nn.Sequential(
            T.Resample(orig_freq=44100, new_freq=sample_rate),
            TimeShift(shift_max=0.2),
            AddGaussianNoise(noise_level=0.005),
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=2048,
                hop_length=512,
                n_mels=128,
                f_min=50,
                f_max=sample_rate // 2
            ),
            T.AmplitudeToDB(),
            SpectrogramPitchShift(max_shift_bins=5, p=0.5),
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
