import h5py
import numpy as np


def load_data(feature_name, target_name, file):
    dataset = h5py.File(file, 'r')

    x_data = np.array(dataset[feature_name])

    y_data = []
    for n in target_name:
        y_data.append(np.array(dataset[n]))
    y_data = np.array(y_data).T

    dataset.close()

    return x_data, y_data


def instance_diff_max(data):
    for i in range(data.shape[0]):
        m = data[i].max()
        data[i] = data[i] / m
    return data


def tilt_sin(y_data):
    st = []
    for i in range(3):
        st.append(y_data[:, i] * np.sin(y_data[:, i + 3]))

    return np.array(st)


def tilt_cos(y_data):
    ct = []
    for i in range(3):
        ct.append(y_data[:, i] * np.cos(y_data[:, i + 3]))

    return np.array(ct)


def tilt_cos_sin(y_data):
    return np.transpose(np.concatenate([tilt_sin(y_data), tilt_cos(y_data)])) * (1 / 0.6)


def tilt_mult_angle(y_train):
    return tilt_cos_sin(y_train)
