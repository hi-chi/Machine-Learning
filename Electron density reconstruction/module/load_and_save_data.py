import os

import numpy as np
import torch


def load_for_3d(path):
    data = np.load(path)
    return (
        data['i1'], data['i2'],
        data['spectrum'], data['angle_x'],
        data['angle_y'], data['angle_dist']
        )

def save_model(model, optimizer, file):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, file)

def load_model(model, optimizer, file):
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer


def load_realdata(base_dir):
    r1 = []
    r2 = []
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    num_subdirs = len(subdirs)

    for idx in range(num_subdirs):
        dir_path = os.path.join(base_dir, str(idx))
        i1_path = os.path.join(dir_path, 'i1.npy')
        i2_path = os.path.join(dir_path, 'i2.npy')

        if os.path.exists(i1_path) and os.path.exists(i2_path):
            r1.append(np.load(i1_path))
            r2.append(np.load(i2_path))

    return np.array(r1), np.array(r2)