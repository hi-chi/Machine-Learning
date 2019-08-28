

"""Neural networks training and prediction of result

This module helps train fully connected neural network and predict result
using the keras library. The functions train() and predict() provide users
access to module.The main functionality for working with the library
is implemented in functions _train() and _predict().
"""


import h5py
import numpy as np

def _train(x_train, y_train, num_layers, layer_dims, output_file):
    """train a fully connected neural network

        Parameters
    ----------
    x_train : ndarray
        data set for traning neural network
    y_train : ndarray
        tags set for traning neural network
    num_layers : integer
        number of layers
    layer_dims : list
        network layer dimension
    output_file : string
        a line containing the directory and file name for saving
        the trained model
    """
    from keras.layers import Dense
    from keras.models import Sequential
    from keras import losses as kl
    from keras import optimizers as opts

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    model = Sequential()
    model.add(Dense(layer_dims[0], input_dim=input_dim, activation='relu'))
    for layer in range(1, num_layers):
        model.add(Dense(layer_dims[layer], activation='relu'))
    model.add(Dense(output_dim))
    opt = opts.Adam()

    model.compile(loss=kl.mean_squared_error, optimizer=opt, metrics=['mean_squared_error'])

    model.fit(x_train, y_train, epochs=25, batch_size=128, shuffle=True)
    model.save(output_file)

def _predict(model_file, x_test, output_file):
    """predict the result

        Parameters
    ----------
    model_file : string
        a line containing the directory and file name for loading
        the model
    x_test : ndarray
        data set for which it is necessary to predict the result
    output_file : string
        line containing the directory and file name to save the result
    """

    from keras.models import load_model
    model = load_model(model_file)
    y_test_pred = model.predict(x_test)
    h5f = h5py.File(output_file, 'w')
    h5f.create_dataset('y', data=y_test_pred)
    h5f.close()

def train(input_file, num_layers, layer_dims, model_file):
    """outer shell for the _train() function.
    reads data and tags from the input file

        Parameters
    ----------
    input_file : string
        a string containing the directory and name of the file storing
        data and tags for training
    num_layers : integer
        number of layers
    layer_dims : list
        network layer dimension
    model_file : string
        a line containing the directory and file name for saving
        the trained model
    """
    file = h5py.File(input_file, 'r')

    x_train = np.asarray(file['x'])
    y_train = np.asarray(file['y'])

    _train(x_train, y_train, num_layers, layer_dims, model_file)

def predict(input_file, model_file, output_file):
    """outer shell for the _predict() function.
    reads data from the input file.

        Parameters
    ----------
    input_file : string
        a string containing the directory and file name of the data storage to predict the result
    model_file : string
        a line containing the directory and file name for loading
        the model
    output_file : string
        line containing the directory and file name to save the result
    """
    file = h5py.File(input_file, 'r')
    x_test = np.asarray(file['x'])
    _predict(model_file, x_test, output_file)
