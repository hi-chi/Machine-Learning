

"""Generatе a random data set.

Data describing a unified theory for training a neural network
to restore the initial set of parameters for the distribution
of droping balls in the Galton board.
"""


import math
import numpy as np
import h5py


def u_distribution(N, j, p):
    tmp = p[0] * math.sqrt(8 / (math.pi * N)) * math.exp(-p[1] * (j - N/2)**2 - p[2] * 0.1 * (j - N/2)**4)
    return tmp

def noise(y_clean):
    """Setting of noise on tags is used to improve learning"""
    random_value = np.random.sample(3)
    y_noise = y_clean * (1 - 0.005 + 0.01 * random_value)
    return y_noise

def save(file, x_train, y_train):
    """Save data set"""
    h5f = h5py.File(file, 'w')
    h5f.create_dataset('x', data=x_train)
    h5f.create_dataset('y', data=y_train)
    h5f.close()

def genereta_data(size_train, number_bins):
    """Generates a data set for network training"""
    x_train = np.zeros(shape=(size_train, number_bins))
    y_train = np.zeros(shape=(size_train, 3))

    for i in range(size_train):
        y_train[i] = np.random.sample(3)
        for j in range(number_bins):
            x_train[i][j] = u_distribution(number_bins, j, y_train[i])
        noise(y_train[i])

    return x_train, y_train

def main():
    """Main function for script training_data.py.

    Setting basic parameters, starting data generation and saving.
    """
    size_train = 400000
    number_bins = 128
    file = '../data/train.h5'
    x_train, y_train = genereta_data(size_train, number_bins)

    save(file, x_train, y_train)

if __name__ == "__main__":
    main()
