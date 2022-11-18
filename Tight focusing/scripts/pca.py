import h5py
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses as kl
from tensorflow.keras import optimizers as opts

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import utils

def split_data(x_data, y_data, random_state):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
                                        x_data, y_data,
                                        test_size=0.15,
                                        random_state=random_state)

    return x_train, y_train,  x_test, y_test

if not os.path.exists('../result/'):
    os.mkdir('../result/')
if not os.path.exists('../result/pca/'):
    os.mkdir('../result/pca/')
hdf = '../data/spectrum.h5'

file =  h5py.File(hdf, 'r')

spectra = np.array(file['spectrum'])
phi = np.array(file['phi'])
theta = np.array(file['theta'])
lp = np.array(file['lp'])

target = np.concatenate([phi.reshape(-1, 1),
                         theta.reshape(-1, 1),
                         lp.reshape(-1, 1)], axis=1)

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

    history = model.fit(x_train, y_train, epochs=epochs,
                        batch_size=128, validation_split=0.0,
                        shuffle=True, verbose=0)

    return model, history

k = 1
epochs = 300

path = '../result/pca/'

seed = [625, 792, 625, 116, 553, 724, 369, 133, 647, 570]

y_min = np.array([0, 0, 0.08])
y_max = np.array([180, 75, 0.78])

for i in range(k):
    np.random.seed(seed[i])
    tf.random.set_seed(seed[i])

    x_train, y_train, x_test, y_test = split_data(spectra.copy(), target.copy(), seed[i])

    scaler = StandardScaler()

    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    pca = PCA(n_components=40)
    pca.fit(x_train)

    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    x_train, x_test, x_min, x_max = utils.preprocessing(x_train, x_test)
    y_train, y_test, y_min, y_max = utils.preprocessing(y_train, y_test)

    model, history = train_nn(x_train, y_train, epochs)

    model.save(path + 'model' + str(i) +'.h5')
    utils.save_data(x_test, y_test, i, path)
