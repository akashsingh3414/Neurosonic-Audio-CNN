import torch
from torch.utils.data import Dataset
import pandas as pd
import soundfile as sf
from pathlib import Path

class ESC50Dataset(Dataset):
    """Enhanced ESC-50 dataset with flexible fold selection"""
    def __init__(self, data_dir, metadata_file, train_folds=None, test_fold=None, 
                 split="train", transform=None, classes=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        # Flexible fold selection for cross-validation
        if train_folds is not None:
            self.metadata = self.metadata[self.metadata['fold'].isin(train_folds)]
        elif test_fold is not None:
            self.metadata = self.metadata[self.metadata['fold'] == test_fold]
        else:
            # Default: folds 1-4 for train, 5 for validation
            if split == 'train':
                self.metadata = self.metadata[self.metadata['fold'] != 5]
            else:
                self.metadata = self.metadata[self.metadata['fold'] == 5]

        # Class mapping
        if classes is None:
            self.classes = sorted(self.metadata['category'].unique())
        else:
            self.classes = classes
                
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        audio_path = Path(self.data_dir) / "audio" / row['filename']
        
        # Load audio
        waveform, sr = sf.read(str(audio_path))
        waveform = torch.from_numpy(waveform).unsqueeze(0).float()

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Apply transforms
        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, torch.tensor(row['label'], dtype=torch.long)
