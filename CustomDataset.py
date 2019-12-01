import torchvision.transforms as transforms
from utils import pad_dimesions
from spectogram import pretty_spectrogram
from scipy.io import wavfile
import numpy as np
import torch

### Parameters ###
fft_size = 2048 # window size for the FFT
step_size = fft_size/16 # distance to slide along the window (in time)
spec_thresh = 4 # threshold for spectrograms (lower filters out more noise)
lowcut = 500 # Hz # Low cut for our butter bandpass filter
highcut = 15000 # Hz # High cut for our butter bandpass filter
# For mels
n_mel_freq_components = 64 # number of mel frequency channels
shorten_factor = 10 # how much should we compress the x-axis (time)
start_freq = 300 # Hz # What frequency to start sampling our melS from 
end_freq = 8000 # Hz # What frequency to stop sampling our melS from

class CustomDatasetSimple():
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

        rate, data = wavfile.read(self.X[idx])
        spec = pretty_spectrogram(data.astype('float64'), fft_size = fft_size, step_size = step_size, log = True, thresh = spec_thresh)
        height = spec.shape[0]
        if height!=112:
            spec = pad_dimesions(spec)

        X = np.copy(spec)
        X = torch.tensor(X, dtype=torch.float)
        X.unsqueeze_(-1)
        X = X.transpose(2, 0)
        X = X.transpose(2, 1)

        X_batch = (X-self.mean)/self.std
        y_batch = self.y[idx]
        
        return X_batch, y_batch