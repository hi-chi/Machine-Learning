"""This script runs simulations for each a00 value.

This data will later be used to build a histogram
for training and testing machine learning models.

Example:
    python generating_data.py
"""

import numpy as np
import h5py
from tqdm import tqdm
import pyHiChi as pfc

import beam_generation as bg
import setting_fields as sf


def save(file, x_data, y_data):
    """Saving data in file.

    Args:
        file: name file in format .h5
        x_data: features
        y_data: label

    """

    h5f = h5py.File(file, 'w')

    h5f.create_dataset('e', data=x_data)
    h5f.create_dataset('y', data=y_data)

    h5f.close()


def solver(a00, step_t, number_electron):
    """Simulate an experiment.

    Args:
        a00: parameter to be predicted
        step_t: time step
        number_electron: the number of electrons for the experiment

    Returns:
        ensemble of electrons after simulation
    """

    c = pfc.LIGHT_VELOCITY

    leight_sm = bg.LIGHT_FS * 1e-15 * c

    gamma = 1000

    grid_size = pfc.Vector3d(2, 2, 2)
    min_coords = pfc.Vector3d(-leight_sm, -leight_sm, -leight_sm)
    max_coords = pfc.Vector3d(leight_sm, leight_sm, leight_sm)
    steps_grid = sf.step(min_coords, max_coords, grid_size)

    sf.a0 = a00
    time_step = (10 ** -4 / (c * step_t))
    func = sf.gen_func()
    grid = pfc.YeeField(grid_size, min_coords, steps_grid, time_step)

    ensemble = bg.gen_beam(number_electron, gamma)
    grid.analytical(func[0], func[1], func[2], func[3], func[4], func[5])

    qed = pfc.QED()

    for i in range(0, step_t):
        grid.set_time(i * time_step)
        qed.process_particles(ensemble, grid, time_step)

    return ensemble


def calculation_gamma(ensemble):
    """Energy calculation.

    Energy is normalized to the speed of light multiplied by the mass of an electron
    for the convenience of further processing.

    Args:
        ensemble: ensemble of electrons after simulation

    Returns:
        normalized energy
    """
    el = ensemble[pfc.ELECTRON]
    gammas_electron = np.zeros([el.size()], dtype=np.float64)

    for num in range(el.size()):
        gammas_electron[num] = np.sqrt((pfc.LIGHT_VELOCITY * pfc.ELECTRON_MASS) ** 2
                                       + el[num].get_momentum().norm2()) / \
                               (pfc.LIGHT_VELOCITY * pfc.ELECTRON_MASS)

    return gammas_electron


def main():
    """Main function of generating data for machine learning.

    Running lots of simulations of the experiment. The calculation of the
    normalized energy for every simulation. Saving data
    for all the necessary values of the parameter a00.
    """
    step_t = 100
    number_electrons = 10000

    range_ = range(10, 1000, 10)

    my_len = len(range_)

    gamma = np.zeros((my_len, number_electrons))
    y_data = np.zeros((my_len, 1))

    for i, a00 in tqdm(enumerate(range_)):
        ensemble = solver(a00, step_t, number_electrons)

        gamma[i] = calculation_gamma(ensemble)
        np.random.shuffle(gamma[i])

        y_data[i] = a00

    save('../data/gamma_electrons.h5', gamma, y_data)


if __name__ == "__main__":
    main()
