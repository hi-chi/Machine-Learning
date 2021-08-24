import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from keras.models import load_model

import utils


def load_data(file):
    dataset = h5py.File(file, 'r')

    x_test = np.array(dataset['x_test'])
    y_test = np.array(dataset['y_test'])

    dataset.close()

    return x_test, y_test


models = {}
accuracy = {}

os.mkdir('../result/pca_enumeration/')
path = '../result/pca_enumeration/'

y_min = np.array([0, 0, 0.08])
y_max = np.array([180, 75, 0.78])

for i in range(100, 2, -10):
    acc = []
    mod = []
    for j in range(3):
        mod.append(load_model(path + 'model' + str(i) + '_' + str(j) + '.h5'))
        x, y = load_data(path + 'data' + str(i) + '_' + str(j) + '.h5')

        y_pred = mod[j].predict(x)
        y_pred, y = utils.postprocessing(y_pred, y, y_min, y_max)
        acc.append(utils.metric(utils.create_df(y, y_pred)))

    accuracy[i] = acc

result = []
met = 'average'
crit = 'MRE'

for k in accuracy.keys():
    result.append([accuracy[k][0][met][crit], accuracy[k][1][met][crit], accuracy[k][2][met][crit]])

df_accuracy = pd.DataFrame(result, index=list(accuracy.keys()))
df_accuracy = df_accuracy.T
print(df_accuracy.mean(axis=0))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

ax.plot(list(accuracy.keys()), list(df_accuracy.mean(axis=0).values))
ax.scatter(list(accuracy.keys()), list(df_accuracy.mean(axis=0).values))

ax.set_xlabel('Number of components', fontsize=14)
ax.set_ylabel('Mean relative error', fontsize=14)

plt.tight_layout()
plt.grid()
plt.savefig('../picture/pca_enumeration.png')
