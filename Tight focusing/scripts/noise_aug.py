import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['TF_NUM_INTEROP_THREADS'] = '3'
os.environ['TF_NUM_INTRAOP_THREADS'] = '3'

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

if not os.path.exists('../result/'):
    os.mkdir('../result/')
if not os.path.exists('../result/noise_aug/'):
    os.mkdir('../result/noise_aug/')
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

def compile_(x_train):
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

    return model

def lr_scheduler_clean(epoch, lr):
    if epoch > 300:
        return lr * 0.99
    return lr

def lr_scheduler_noise(epoch, lr):
    if epoch > 300:
        return lr * 0.99
    return lr

def train_clean(model, x_train, y_train, epoch, initial_epoch):
    callbacks = [LearningRateScheduler(lr_scheduler_clean)]
    history = model.fit(x_train, y_train, epochs=epoch, initial_epoch=initial_epoch, callbacks=callbacks,
                        batch_size=128, validation_split=0.0, shuffle=True,
                        verbose=0)

    return model, history

def train_noise(model, x_train, y_train, epoch, initial_epoch):
    callbacks = [LearningRateScheduler(lr_scheduler_noise)]
    history = model.fit(x_train, y_train, epochs=epoch, initial_epoch=initial_epoch, callbacks=callbacks,
                        batch_size=128, validation_split=0.0, shuffle=True,
                        verbose=0)

    return model, history

def split_data(spectra, target, noise_estimation, train_idx, test_idx, num_noise):
    xdata = []
    ydata = []
    noise_data = []

    for _ in range(num_noise):
        noise_data.append(noise_estimation[train_idx])
        xdata.append(spectra[train_idx])
        ydata.append(target[train_idx])

    x_train = np.concatenate(xdata)
    y_train = np.concatenate(ydata)
    noise_data = np.concatenate(noise_data)

    x_test = spectra[test_idx]
    y_test = target[test_idx]

    return x_train, y_train, x_test, y_test, noise_data

noise_estimation = utils.get_std(spectra.copy(), target.copy())

path = '../result/noise_aug/'
seed = [920, 814, 809, 235, 201, 120, 936, 47, 334, 253]

k = 10
epochs = 400

num_noise = 5
coef_noise = 1
num_step = 5
noise_epoch = 200

for i in range(5, k):
    np.random.seed(seed[i])
    tf.random.set_seed(seed[i])
    idxs = np.random.permutation(spectra.shape[0])

    num_train_samples = int(0.85 * spectra.shape[0])
    num_test_samples = spectra.shape[0] - num_train_samples
    train_idxs = idxs[:num_train_samples]
    test_idxs = idxs[num_train_samples:]

    x_train = spectra[train_idxs]
    y_train = target[train_idxs]

    x_test = spectra[train_idxs]
    y_test = target[train_idxs]

    x_train, x_test, x_min, x_max = utils.preprocessing(x_train, x_test)
    y_train, y_test, y_min, y_max = utils.preprocessing(y_train, y_test)

    model = compile_(x_train)
    model, history = train_clean(model, x_train, y_train, epoch=epochs, initial_epoch=0)

    for j in range(num_step):
        sh = coef_noise / num_step
        h = noise_epoch / num_step

        x_train, y_train, x_test, y_test, noise_train = split_data(spectra.copy(), target.copy(),
                                                        noise_estimation, train_idxs, test_idxs, num_noise)
        if j == num_step - 1:
            x_train = utils.add_noise(x_train, noise_train, sh*(j+1))
            x_test = utils.add_noise(x_test, noise_estimation[test_idxs], sh*(j+1))
            x_train, x_test, x_min, x_max = utils.preprocessing(x_train, x_test)
            y_train, y_test, y_min, y_max = utils.preprocessing(y_train, y_test)

        else:
            x_train = add_noise(x_train, noise_train, sh*(j+1))
            x_train, _, x_min, x_max = utils.preprocessing(x_train, x_test)
            y_train, _, y_min, y_max = utils.preprocessing(y_train, y_test)

        model, history = train_noise(model, x_train, y_train, epoch=int((j+1)*h), initial_epoch=int(j*h))

        if j == num_step - 1:
            model.save(path + 'model' + str(i) +'.h5')
            utils.save_data(x_test, y_test, i, path)
