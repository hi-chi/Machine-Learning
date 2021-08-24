import h5py
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'

import tensorflow as tf

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses as kl
from tensorflow.keras import optimizers as opts
from keras.callbacks import LearningRateScheduler

import utils

if not os.path.exists('../result/'):
    os.mkdir('../result/')
if not os.path.exists('../result/aug/'):
    os.mkdir('../result/aug/')
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

target = np.concatenate([phi.reshape(-1, 1),
                         theta.reshape(-1, 1),
                         lp.reshape(-1, 1)], axis=1)

def lr_scheduler(epoch, lr):
    if epoch > 150:
        return lr * 0.95
    return lr

def train_nn(x_train, y_train, epoch):

    model = Sequential()

    model.add(Dense(1024, input_dim = x_train.shape[1], activation = 'relu'))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))

    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))

    model.add(Dense(3, activation = 'linear'))

    opt = opts.Adam(lr = 0.0005, amsgrad=False)
    model.compile(loss=kl.mean_squared_error,
                optimizer=opt)

    callbacks = [LearningRateScheduler(lr_scheduler)]
    history = model.fit(x_train, y_train, epochs=epoch, callbacks=callbacks,
                        batch_size=128, shuffle=True, verbose=0)
    return model, history

def add_noise(data, coef=1):
    noise = data.copy()

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            noise[i][j] += np.random.normal(0, np.abs(noise[i][j]*coef), size=1)[0]

    return noise

def split_data(spectra, target, train_idx, test_idx, num_noise, coef_noise):
    xdata = []
    ydata = []

    for _ in range(num_noise):
        xdata.append(add_noise(spectra[train_idx], coef_noise))
        ydata.append(target[train_idx])

    x_train = np.concatenate(xdata)
    y_train = np.concatenate(ydata)

    x_test = spectra[test_idx]
    y_test = target[test_idx]

    return x_train, y_train, x_test, y_test


path = '../result/aug/'

k = 10
epochs = 200

num_noise = 5
coef_noise = 0.02

seed = [270, 754, 179, 734, 379, 625, 663, 747, 227, 455]

for i in range(6, k):
    np.random.seed(seed[i])
    tf.random.set_seed(seed[i])

    idxs = np.random.permutation(spectra.shape[0])

    num_train_samples = int(0.85 * spectra.shape[0])
    num_test_samples = spectra.shape[0] - num_train_samples
    train_idxs = idxs[:num_train_samples]
    test_idxs = idxs[num_train_samples:]

    x_train, y_train, x_test, y_test = split_data(spectra.copy(), target.copy(),
                                                  train_idxs, test_idxs,
                                                  num_noise, coef_noise)

    x_train, x_test, x_min, x_max = utils.preprocessing(x_train, x_test)
    y_train, y_test, y_min, y_max = utils.preprocessing(y_train, y_test)

    model, history = train_nn(x_train, y_train, epoch=epochs)

    model.save(path + 'model' + str(i) +'.h5')
    utils.save_data(x_test, y_test, i, path)
