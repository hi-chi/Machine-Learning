"""This script builds histograms.

A sample without repetitions selects the number of electrons
equal to n_bins * n_electrons * (N_TRAIN_DATA + 2) for each а00.
N_TRAIN_DATA times data for each а00. Once for validation and test samples

The script uses command line arguments. Use --help for more information.

Example:
    python create_histogram.py -if electrons.h5 -of data_xgb.h5 -nb 20 -ne 500
"""

import sys
import argparse
import numpy as np
import h5py


N_TRAIN_DATA = 3


def create_parser():
    """Parsing command line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument('-if', '--in_file', default="data.h5", type=str,
                        help="name of the source data file (in format .h5)")
    parser.add_argument('-of', '--out_file', default="data.h5", type=str,
                        help="file name with histograms data (in format .h5)")

    parser.add_argument('-nb', '--n_bins', default=10, type=int,
                        help="the number of bins in the histogram")
    parser.add_argument('-ne', '--n_electrons', default=10, type=int,
                        help="the number of electrons per bin on average")
    return parser


def read(file):
    """Reading data from files.

    Args:
        file: name file in format .h5

    Returns:
        array of electrons, array of labels
    """

    h5f = h5py.File(file, 'r')

    electrons = np.asarray(h5f['electrons'])
    y_label = np.asarray(h5f['y'])

    h5f.close()

    return electrons, y_label


def save(file, x_train, y_train, x_test, y_test, x_valid, y_valid):
    """Saving data in file.

    Args:
        file: name file in format .h5

    Returns:
        data arrays to save
    """

    h5f = h5py.File(file, 'w')

    h5f.create_dataset('x_train', data=x_train)
    h5f.create_dataset('y_train', data=y_train)

    h5f.create_dataset('x_test', data=x_test)
    h5f.create_dataset('y_test', data=y_test)

    h5f.create_dataset('x_valid', data=x_valid)
    h5f.create_dataset('y_valid', data=y_valid)

    h5f.close()


def create_hist(gamma_electrons, y_label, n_bins, n_electrons):
    """Creating an array that stores the bin values of the histogram
    by slices from the original data with the electron energy.

    Args:
        gamma_electrons: electron energy
        y_label: label
        n_bins: the number of bins in the histogram
        n_electrons: the number of electrons per bin on average

    Returns:
        histograms
    """

    x_train = np.zeros(shape=(y_label.shape[0] * N_TRAIN_DATA, n_bins))
    y_train = np.zeros(shape=(y_label.shape[0] * N_TRAIN_DATA, 1))

    x_test = np.zeros(shape=(y_label.shape[0], n_bins))
    y_test = np.zeros(shape=(y_label.shape[0], 1))

    x_valid = np.zeros(shape=(y_label.shape[0], n_bins))
    y_valid = np.zeros(shape=(y_label.shape[0], 1))

    slice_size = n_electrons * n_bins

    for i in range(y_label.shape[0]):
        for j in range(N_TRAIN_DATA):
            x_train[i * N_TRAIN_DATA + j], _ = np.histogram(
                gamma_electrons[i][j * slice_size:(j + 1) * slice_size],
                density=True, range=(0, 1000),
                bins=np.linspace(0, 1000, n_bins + 1))

            y_train[i * N_TRAIN_DATA + j] = y_label[i].copy()

    for i in range(y_label.shape[0]):
        x_test[i], _ = np.histogram(
            gamma_electrons[i][N_TRAIN_DATA * slice_size:(N_TRAIN_DATA + 1) * slice_size],
            density=True, range=(0, 1000),
            bins=np.linspace(0, 1000, n_bins + 1))

        y_test[i] = y_label[i].copy()

    for i in range(y_label.shape[0]):
        x_valid[i], _ = np.histogram(
            gamma_electrons[i][(N_TRAIN_DATA + 1) * slice_size:(N_TRAIN_DATA + 2) * slice_size],
            density=True, range=(0, 1000),
            bins=np.linspace(0, 1000, n_bins + 1))

        y_valid[i] = y_label[i].copy()

    return x_train, y_train, x_test, y_test, x_valid, y_valid


def check(data_size, n_bins, n_electrons):
    """Checking whether it is possible to use the requested number of electrons
    to build all histograms.

    In case of lack of electrons in the data file, an exception will be generated.

    Args:
        data_size: the number of electrons in the experiment
        n_bins: the number of bins in the histogram
        n_electrons: the number of electrons per bin on average

    Raises:
        ValueError: If the number of electrons in the file is less
        than the amount of data requested for building histograms
    """

    if n_bins * n_electrons * (N_TRAIN_DATA + 2) > data_size:
        raise ValueError("the requested number of electrons is greater "
                         "than that contained in the data")


def main():
    """The main function of the script.

    Creating a parser, parsing command line arguments,
    creating histograms and saving them.
    """
    parser = create_parser()
    namespace = parser.parse_args(sys.argv[1:])

    electrons, y_label = read(namespace.in_file)

    n_bins = namespace.n_bins
    n_electrons = namespace.n_electrons

    check(electrons.shape[1], n_bins, n_electrons)

    x_train, y_train, x_test, y_test, x_valid, y_valid = create_hist(
        electrons, y_label, n_bins, n_electrons)

    file = namespace.out_file

    save(file, x_train, y_train, x_test, y_test, x_valid, y_valid)


if __name__ == '__main__':
    main()
