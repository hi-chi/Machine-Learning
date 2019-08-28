

"""Use of the NN module to restore Galton board distribution parameters

This script indicates the directories and file names for the correct
operation of the modules. All files must be in hdf5 format. Necessity
to indicate parameters for neural network.

    Files
----------
input_file : int
    a string containing the directory and name of the file storing
    data and tags for training
predict_file : int
    a string containing the directory and file name of the data storage
    to predict the result
model_file : string
    a line containing the directory and file name for loading
    the model
prediction_file : string
    line containing the directory and file name to save the result

    Parameters
----------
num_layers : int
    number of layers
layer_dims : list
    network layer dimension
"""


import NN

input_file = '../data/train.h5'
predict_file = '../data/predict.h5'
model_file = '../model/model.h5'
prediction_file = '../data/result.h5'

num_layers = 4
layer_dims = [128, 64, 32, 16]

NN.train(input_file, num_layers, layer_dims, model_file)
NN.predict(predict_file, model_file, prediction_file)
