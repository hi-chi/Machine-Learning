import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'

import tensorflow as tf

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses as kl
from tensorflow.keras import optimizers as opts

import utils


def split_data(x_data, y_data):
    x_train, x_test, y_train, y_test = train_test_split(
                                        x_data, y_data, test_size=0.15)

    return x_train, y_train,  x_test, y_test

hdf = '../data/spectrum.h5'

file =  h5py.File(hdf, 'r')

spectra = np.array(file['spectrum'])
phi = np.array(file['phi'])
theta = np.array(file['theta'])
lp = np.array(file['lp'])

target = np.concatenate([phi.reshape(-1, 1), theta.reshape(-1, 1), lp.reshape(-1, 1)], axis=1)

def train_nn(x_train, y_train, epochs):

    model = Sequential()

    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))

    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))

    model.add(Dense(3, activation = 'linear'))

    opt = opts.Adam(lr = 0.0005)
    model.compile(loss=kl.mean_squared_error,
                optimizer=opt)

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=128,
        validation_split=0.0, shuffle=True, verbose=0)

    return model, history

epochs = 300

accuracy = {}

path = '../result/pca_enumeration/'

np.random.seed(42)
tf.random.set_seed(42)

for i in range(100, 2, -10):
    acc = []
    for j in range(3):
        x_train, y_train, x_test, y_test = split_data(spectra.copy(), target.copy())

        scaler = StandardScaler()
        scaler.fit(x_train)

        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        pca = PCA(n_components=i)
        pca.fit(x_train)

        x_train_pca = pca.transform(x_train)
        x_test_pca = pca.transform(x_test)

        x_train_pca, x_test_pca, x_min, x_max = utils.preprocessing(x_train_pca, x_test_pca)
        y_train, y_test, y_min, y_max = utils.preprocessing(y_train, y_test)

        model, history = train_nn(x_train_pca, y_train, epochs)

        y_pred = model.predict(x_test_pca)

        model.save(path + 'model' + str(i) + '_' + str(j) +'.h5')
        utils.save_data(x_test_pca, y_test, str(i) + '_' + str(j), path)

    accuracy[i] = acc
