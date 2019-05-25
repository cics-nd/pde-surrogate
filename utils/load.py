"""
Load args and model from a directory
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from argparse import Namespace
import h5py
import json


def load_args(run_dir):
    with open(run_dir + '/args.txt') as args_file:  
        args = Namespace(**json.load(args_file))
    # pprint(args)
    return args


def load_data(hdf5_file, ndata, batch_size, only_input=True, return_stats=False):
    with h5py.File(hdf5_file, 'r') as f:
        x_data = f['input'][:ndata]
        print(f'x_data: {x_data.shape}')    
        if not only_input:
            y_data = f['output'][:ndata]
            print(f'y_data: {y_data.shape}')    

    stats = {}
    if return_stats:
        y_variation = ((y_data - y_data.mean(0, keepdims=True)) ** 2).sum(
            axis=(0, 2, 3))
        stats['y_variation'] = y_variation
    
    data_tuple = (torch.FloatTensor(x_data), ) if only_input else (
            torch.FloatTensor(x_data), torch.FloatTensor(y_data))
    data_loader = DataLoader(TensorDataset(*data_tuple),
        batch_size=batch_size, shuffle=True, drop_last=True)
    print(f'Loaded dataset: {hdf5_file}')
    return data_loader, stats

