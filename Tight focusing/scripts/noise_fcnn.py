import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import h5py
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses as kl
from tensorflow.keras import optimizers as opts
from keras.callbacks import LearningRateScheduler

import utils

def split_data(x_data, y_data, seed):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
                                        x_data, y_data, test_size=0.15, random_state=seed)

    return x_train, y_train,  x_test, y_test

if not os.path.exists('../result/'):
    os.mkdir('../result/')
if not os.path.exists('../result/noise_fcnn/'):
    os.mkdir('../result/noise_fcnn/')
hdf = '../data/spectrum.h5'

file =  h5py.File('../data/spectrum1.h5', 'r')

spectra1 = np.array(file['spectrum'])
phi1 = np.array(file['phi'])
theta1 = np.array(file['theta'])
lp1 = np.array(file['lp'])

file =  h5py.File('../data/spectrum2.h5', 'r')

spectra2 = np.array(file['spectrum'])
phi2 = np.array(file['phi'])
theta2 = np.array(file['theta'])
lp2 = np.array(file['lp'])

spectra = np.concatenate([spectra1, spectra2])
phi = np.concatenate([phi1, phi2])
theta = np.concatenate([theta1, theta2])
lp = np.concatenate([lp1, lp2])

target = np.concatenate([phi.reshape(-1, 1), theta.reshape(-1, 1), lp.reshape(-1, 1)], axis=1)

def lr_scheduler(epoch, lr):
    if epoch > 300:
        return lr * 0.99
    return lr

def train_nn(x_train, y_train, epochs):

    model = Sequential()

    model.add(Dense(1024, input_dim = x_train.shape[1], activation = 'relu'))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))

    model.add(Dense(3, activation = 'linear'))

    opt = opts.Adam(lr = 0.0005)
    model.compile(loss=kl.mean_squared_error,
                optimizer=opt)

    callbacks = [LearningRateScheduler(lr_scheduler)]

    history = model.fit(x_train, y_train, epochs=epochs, callbacks=callbacks,
                        batch_size=128, validation_split=0.0, shuffle=True,
                        verbose=0)

    return model, history

def add_noise(data, sigma):
    noise = data.copy()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            noise[i][j] += np.random.normal(0, sigma[i][j], size=1)[0]

    return noise

noise_estimation = utils.get_std(spectra.copy(), target.copy())

path = '../result/noise_fcnn/'

k = 10
epochs = 400
seed = [895, 342, 664, 494, 128, 467, 573, 569, 574, 379]

for i in range(6, k):
    np.random.seed(seed[i])
    tf.random.set_seed(seed[i])

    spectra_noise = add_noise(spectra.copy(), noise_estimation.copy()) # ???
    x_train, y_train, x_test, y_test = split_data(spectra_noise.copy(), target.copy(), seed[i])

    x_train, x_test, x_min, x_max = utils.preprocessing(x_train, x_test)
    y_train, y_test, y_min, y_max = utils.preprocessing(y_train, y_test)

    model, history = train_nn(x_train, y_train, epochs)

    model.save(path + 'model' + str(i) +'.h5')
    utils.save_data(x_test, y_test, i, path)
