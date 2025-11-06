from functools import partial
from multiprocessing import Pool

import numpy as np

from module.constant import data_config
from src.data_gen.numerical_model import tsms11, tsms18
from src.data_gen.utils.pygenerators import get_samples, Generator3DBeam
from src.preprocessing.data3d import resize_3d_data

def num(distribution, N, key, shapes):
    data_config.set_global_key(key)
    distribution = resize_3d_data(distribution[0], new_size=shapes)

    samples = get_samples(distribution, N)
    pgen = Generator3DBeam(samples, data_config['ANGLE_COEFFICIENT'],  data_config['BASELINE_ANGLE'])

    if data_config['name'] == 'tsms11':
        simulation = tsms11.simulation
    if data_config['name'] == 'tsms18':
        simulation = tsms18.simulation
    else:
        ValueError()

    screen1, screen2 = simulation(pgen)

    return screen1, screen2

def run_num_simulation(distribution, N, shapes):
    with Pool() as pool:
        results = pool.map(partial(num, N=N, key=data_config.global_key, shapes=shapes), distribution)

    screen1, screen2 = zip(*results)
    screen1 = np.array(screen1)
    screen2 = np.array(screen2)

    return screen1, screen2
