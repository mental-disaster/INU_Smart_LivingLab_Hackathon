import os
import sys
import numpy as np
import tensorflow as tf
import random

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config.config import get_config

config = get_config()

def prepare_data(data_path, ctrl):
    with open(ctrl, 'r') as f:
        files = f.read().splitlines()

    data_shuffle, labels_shuffle = [], []
    data_range = list(range(len(files)))
    random.shuffle(data_range)

    for s in data_range:
        filename = files[s] + '.mfc'
        load_path = os.path.join(data_path, filename)
        load_feat = np.loadtxt(load_path)

        # fix_frame process - Can be modified
        if load_feat.shape != (config.mfc*3, config.n_frame):
            load_feat = load_feat[:, :config.n_frame]

        data_shuffle.append(load_feat)
        label = files[s].split('_')[2]
        labels_shuffle.append(int(label))

    labels_shuffle = np.eye(config.n_class)[labels_shuffle]
        
    return tf.expand_dims(np.asarray(data_shuffle), 3), np.asarray(labels_shuffle)