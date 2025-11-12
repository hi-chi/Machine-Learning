import os
import sys

from os import listdir
from os.path import isfile, join

import numpy as np
from tqdm import tqdm

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from module.lazy_array import LazyArray

def convert(directory):
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

    i1 = LazyArray(len(onlyfiles), dtype=np.float32)
    i2 = LazyArray(len(onlyfiles), dtype=np.float32)
    spectrum = LazyArray(len(onlyfiles), dtype=np.float32)
    angle_x = LazyArray(len(onlyfiles), dtype=np.float32)
    angle_y = LazyArray(len(onlyfiles), dtype=np.float32)
    angle_dist = LazyArray(len(onlyfiles), dtype=np.float32)

    for i, file in enumerate(tqdm(onlyfiles)):
        with np.load(directory + file) as data:
            i1_ = data['i1'].reshape(-1, *data['i1'].shape)
            i2_ = data['i2'].reshape(-1, *data['i2'].shape)

            i1[i] = i1_
            i2[i] = i2_
            spectrum[i] = data['spectrum'].reshape(-1, *data['spectrum'].shape)
            angle_x[i] = data['angle_x'].reshape(-1, *data['angle_x'].shape)
            angle_y[i] = data['angle_y'].reshape(-1, *data['angle_y'].shape)
            angle_dist[i] = data['angle_dist'].reshape(-1, *data['angle_dist'].shape)

    i1 = i1.array
    i2 = i2.array
    spectrum = spectrum.array
    angle_x = angle_x.array
    angle_y = angle_y.array
    angle_dist = angle_dist.array

    return i1, i2, spectrum, angle_x, angle_y, angle_dist


def save_convert(path, i1, i2, spectrum, angle_x, angle_y, angle_dist):
    np.savez_compressed(
        path,
        i1=i1,
        i2=i2,
        spectrum=spectrum,
        angle_x=angle_x,
        angle_y=angle_y,
        angle_dist=angle_dist,
    )


if __name__ == '__main__':
    os.makedirs('../../../datasets/numerical_dataset/processed/', exist_ok=True)

    name = 'tsms11_train_50000'

    i1, i2, spectrum, angle_x, angle_y, angle_dist = convert(
        directory=f'../../../datasets/numerical_dataset/raw/{name}/'
    )
    save_convert(
        f'../../../datasets/numerical_dataset/processed/{name}',
        i1, i2, spectrum, angle_x, angle_y, angle_dist
    )