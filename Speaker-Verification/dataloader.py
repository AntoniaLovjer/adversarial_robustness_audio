import os
import re
from glob import glob
import torch
import pandas as pd

WORDS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(WORDS)}
name2id = {name: i for i, name in id2name.items()}

def load_data(data_dir):
    """ Return 2 lists of tuples:
    [(user_id, class_id, path), ...] for train
    [(user_id, class_id, path), ...] for validation
    """
    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")

    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

    possible = set(WORDS)
    data = []
    for entry in all_files:
        bl_true = True
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'
            

            label_id = name2id[label]

            sample = (uid, label_id, entry)
            data.append(sample)

    data = pd.DataFrame(data)
    data.columns = ['uid', 'label_id', 'path']
    return data