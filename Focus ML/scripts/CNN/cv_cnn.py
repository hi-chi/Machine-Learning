import os
import random

import h5py
import numpy as np
import tensorflow as tf
import utils
from sklearn.model_selection import KFold
from tensorflow.keras import losses as kl
from tensorflow.keras import optimizers as opts
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential


FEATURE_NAME = 'flux'
TARGET_NAME = ['tilt0', 'tilt1', 'tilt2', 'angle0', 'angle1', 'angle2']
RESULT_NAME = ['intensity1', 'intensity2', 'tilt0', 'tilt1', 'tilt2', 'cos(angle0)', 'cos(angle1)', 'cos(angle2)',
               'sin(angle0)', 'sin(angle1)', 'sin(angle2)']

FILE_DATA = '../data/paperdata.h5'


def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=42):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def cnn(x_train, y_train, output_layer):
    model = Sequential()

    model.add(
        Convolution2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu', kernel_initializer=HeNormal()))
    model.add(
        Convolution2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu', kernel_initializer=HeNormal()))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(
        Convolution2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', kernel_initializer=HeNormal()))
    model.add(
        Convolution2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', kernel_initializer=HeNormal()))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dense(512, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dense(128, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dense(32, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dense(output_layer, activation='linear'))

    opt = opts.Adam()
    model.compile(loss=kl.mean_squared_error,
                  optimizer=opt)

    history = model.fit(x_train, y_train, epochs=30,
                        batch_size=128, validation_split=0.0, shuffle=True,
                        verbose=0)

    return model, history


def exp_corr_task(x_train, y_train, x_test, y_test, i):
    x_train = utils.instance_diff_max(x_train)
    x_test = utils.instance_diff_max(x_test)

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    y_train = utils.tilt_mult_angle(y_train)
    y_test = utils.tilt_mult_angle(y_test)

    model, history = cnn(x_train, y_train, output_layer=6)
    y_pred = model.predict(x_test)

    with h5py.File(f'../cv/cnn/predict{i}.h5', 'w') as f:
        f.create_dataset("y_test", data=y_test)
        f.create_dataset("y_pred", data=y_pred)


if __name__ == "__main__":
    x_data, y_data = utils.load_data(FEATURE_NAME, TARGET_NAME, FILE_DATA)

    set_global_determinism(seed=42)
    skf = KFold(n_splits=10, shuffle=True, random_state=42)
    for i, (train_idxs, test_idxs) in enumerate(skf.split(x_data, y_data)):
        x_train, y_train = x_data[train_idxs], y_data[train_idxs]
        x_test, y_test = x_data[test_idxs], y_data[test_idxs]

        exp_corr_task(x_train, y_train, x_test, y_test, i)
