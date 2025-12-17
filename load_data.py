# load_data.py
#   Data loading utilities from data/

import pickle

def load_data(data_filename: str, global_batch_size: int):
    with open(data_filename, 'rb') as f:
        data = pickle.load(f)
    print('  [load_data] Loaded', len(data), 'samples from', data_filename)
    usable_len = (len(data) // global_batch_size) * global_batch_size
    return data[:usable_len]  #  data is uint8, (batch, height=128, width=128, channels=3)

def normalize_data(data):
    return data.astype('float32') / 255.0