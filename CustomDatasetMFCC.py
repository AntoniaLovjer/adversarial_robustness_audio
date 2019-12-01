import librosa
import torch
import torch.nn as nnn
from utils import windows

class CustomDatasetMFCC():
    """Simple dataset class for dataloader"""
    def __init__(self, X, y, mean, std):
        """Initialize the CustomDataset"""

        self.mean = mean
        self.std = std
        self.X = X
        self.y = y

    def __len__(self):
        """Return the total length of the dataset"""
        dataset_size = len(self.X)
        return dataset_size

    def __getitem__(self, idx):
        """Return the batch given the indices"""
        bands = 128
        frames = 41
        window_size = 1000
        sound_clip, s = librosa.load(self.X[idx])

        # set the size of the clip to be the same: 22050
        if len(sound_clip) != 22050:
            if len(sound_clip) > 22050:
                sound_clip = soundclip[:22050]
            else:
                tempData = np.zeros(22050)
                tempData[:len(sound_clip)] = sound_clip[:]
                sound_clip = tempData


        # Reset lists for file
        mfccs = []
        for (start,end) in windows(sound_clip, window_size):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T
                mfccs.append(mfcc[1])
        X = torch.tensor(mfccs, dtype=torch.float)
        X.unsqueeze_(-1)
        X = X.transpose(2, 0)
        X = X.transpose(2, 1)
        X_batch = (X - self.mean) / self.std
        y_batch = self.y[idx]
        return X_batch, y_batch