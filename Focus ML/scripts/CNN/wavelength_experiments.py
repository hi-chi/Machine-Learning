import os
import random

import h5py
import numpy as np
import tensorflow as tf
import utils
from tensorflow.keras import losses as kl
from tensorflow.keras import optimizers as opts
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

FEATURE_NAME = 'flux'
TARGET_NAME = ['iwave1', 'iwave2', 'tilt0', 'tilt1', 'tilt2', 'angle0', 'angle1', 'angle2']
RESULT_NAME = ['tilt0_sin0', 'tilt1_sin1', 'tilt2_sin2', 'tilt0_cos0', 'tilt1_cos1', 'tilt2_cos2']


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


def exp(x_train, y_train, x_test, y_test, i, orig_test):
    x_train = utils.instance_diff_max(x_train)
    x_test = utils.instance_diff_max(x_test)

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    y_train = utils.tilt_mult_angle(y_train)
    y_test = utils.tilt_mult_angle(y_test)

    model, history = cnn(x_train, y_train, output_layer=6)

    y_pred = model.predict(x_test)

    with h5py.File(f'../cv/wavelength/predict{i}.h5', 'w') as f:
        f.create_dataset("y_test", data=y_test)
        f.create_dataset("y_pred", data=y_pred)
        f.create_dataset("orig_test", data=orig_test)


if __name__ == "__main__":
    x_data, y_data = utils.load_data(FEATURE_NAME, TARGET_NAME, '../data/train_wavelenght.h5')
    x_test, y_test = utils.load_data(FEATURE_NAME, TARGET_NAME, '../data/paperdata.h5')

    set_global_determinism(seed=42)

    for i in range(10):
        exp(x_data, y_data[:, 2:], x_test, y_test[:, 2:], i, y_test)
