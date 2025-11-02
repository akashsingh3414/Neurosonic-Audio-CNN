import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
from utils.augmentation import NormalizeSpec


class AudioProcessor:
    """Audio preprocessing pipeline for Mel spectrogram conversion."""

    def __init__(self, sample_rate=22050, mel_params=None):
        if mel_params is None:
            mel_params = {
                "n_fft": 2048,
                "hop_length": 512,
                "n_mels": 128,
                "f_min": 50,
                "f_max": sample_rate // 2,
            }

        self.sample_rate = sample_rate
        self.mel_params = mel_params

        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=mel_params["n_fft"],
                hop_length=mel_params["hop_length"],
                n_mels=mel_params["n_mels"],
                f_min=mel_params["f_min"],
                f_max=mel_params["f_max"],
            ),
            T.AmplitudeToDB(),
            NormalizeSpec(),
        )

    def process_audio_chunk(self, audio_data):
        """Convert raw audio array → normalized spectrogram tensor."""
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)
