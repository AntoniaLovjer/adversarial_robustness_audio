import numpy as np
import torch

def log_textfile(filename, text):
    """
    Function log_to_textfile
    
    Appends a text to a file (logs)
    
    Args:
        filename (str): Filename of logfile
        text (str): New information to log (append)
    
    Return:
    
    """
    print(text)
    f = open(filename, "a")
    f.write(str(text) + str('\n'))
    f.close()
    
    
def pad_dimesions(spec):
    '''
    Data comes in several dimensions. Pad with zeros to get dimensions (112,1)
    '''
    x_offset = 1  
    y_offset = 0
    result = np.zeros([112, 1024])
    result[x_offset:spec.shape[0] + x_offset, y_offset:spec.shape[1] + y_offset] = spec
    return result


def get_mean_std(loader):
    output_mean = 0.
    output_std = 0.
    n = 0
    for X,y in loader:
        output_mean += np.mean(X.detach().cpu().numpy())
        output_std += np.std(X.detach().cpu().numpy())
        n += 1
        if n % 10 == 0:
            print(n)
    return output_mean/n, output_std/n


def load_model(name):
    model = torch.load(name)
    return model

def pad_dimesions_mfcc(mfcc):
    '''
    Data comes in several dimensions. Pad with zeros to get dimensions (42, 20)
    '''
    x_offset = 0  
    y_offset = 0
    result = np.zeros([42, 20])
    result[x_offset:mfcc.shape[0] + x_offset, y_offset:mfcc.shape[1] + y_offset] = mfcc
    return result

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2) 
