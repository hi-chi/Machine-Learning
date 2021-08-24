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

pnt = ['phi', 'theta', 'Lp']
pnp = ['phi_pred', 'theta_pred', 'Lp_pred']

def plot_corr_model(y, y_pred, name='plotcorr.png'):
    df = utils.create_df(y, y_pred)
    font_size = 22

    labelx = [r'$\phi_{cep}$', r'$\theta$', r'$L_p$']
    labely = [r'predict $\phi_{cep}$', r'predict $\theta$', r'predict $L_p$']

    for i in range(len(pnt)):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7), dpi=200)

        ax.scatter(df[pnt[i]], df[pnp[i]], s=2)

        red_line = [df[pnt[i]].min(), df[pnt[i]].max()]
        ax.plot(red_line, red_line, color='red')

        ax.set_xlabel(labelx[i], fontsize=font_size)
        ax.set_ylabel(labely[i], fontsize=font_size)

        ax.tick_params(axis='both', which='major', labelsize=15)

        plt.tight_layout()
        plt.savefig('../picture/' + name[:-3] + str(i) + '.png')
        # plt.clf()
        plt.close()


def load_data(file):
    dataset = h5py.File(file, 'r')

    x_test = np.array(dataset['x_test'])
    y_test = np.array(dataset['y_test'])

    dataset.close()

    return x_test, y_test

def get_mean_result(accuracy):
    mean_mre_phi = []
    mean_mre_theta = []
    mean_mre_lp = []

    mean_mre_av = []

    for acc in accuracy:
        mean_mre_phi.append(acc['phi']['MRE'])
        mean_mre_theta.append(acc['theta']['MRE'])
        mean_mre_lp.append(acc['Lp']['MRE'])

        mean_mre_av.append(acc['average']['MRE'])

    mean_r2_phi = []
    mean_r2_theta = []
    mean_r2_lp = []

    mean_r2_av = []

    for acc in accuracy:
        mean_r2_phi.append(acc['phi']['R2'])
        mean_r2_theta.append(acc['theta']['R2'])
        mean_r2_lp.append(acc['Lp']['R2'])

        mean_r2_av.append(acc['average']['R2'])

    data = [[np.mean(mean_mre_phi), np.mean(mean_mre_theta), np.mean(mean_mre_lp), np.mean(mean_mre_av)],
            [np.mean(mean_r2_phi), np.mean(mean_r2_theta), np.mean(mean_r2_lp), np.mean(mean_r2_av)]]
    df_accuracy = pd.DataFrame(np.round(data, 3), columns=accuracy[0].columns, index=["MRE", "R2"])
    return df_accuracy

if not os.path.exists('../picture/'):
    os.mkdir('../picture/')

model_fcnn = []
model_pca = []
model_aug = []

data_fcnn = {}
data_pca = {}
data_aug = {}

accuracy_fcnn = []
accuracy_pca = []
accuracy_aug = []

fcnn_dir = '../result/fcnn/'
pca_dir = '../result/pca/'
aug_dir = '../result/aug/'

k = 10

y_min = np.array([0, 0, 0.08])
y_max = np.array([180, 75, 0.78])

for i in range(k):
    model_fcnn.append(load_model(fcnn_dir + 'model' + str(i) +'.h5'))
    model_pca.append(load_model(pca_dir + 'model' + str(i) +'.h5'))
    model_aug.append(load_model(aug_dir + 'model' + str(i) +'.h5'))

    x, y = load_data(fcnn_dir + 'data' + str(i) + '.h5')
    data_fcnn['x'], data_fcnn['y'] = x, y
    y_pred = model_fcnn[i].predict(x)
    y_pred, y = utils.postprocessing(y_pred, y, y_min, y_max)
    accuracy_fcnn.append(utils.metric(utils.create_df(y, y_pred)))

    x, y = load_data(pca_dir + 'data' + str(i) + '.h5')
    data_pca['x'], data_pca['y'] = x, y
    y_pred = model_pca[i].predict(x)
    y_pred, y = utils.postprocessing(y_pred, y, y_min, y_max)
    accuracy_pca.append(utils.metric(utils.create_df(y, y_pred)))

    x, y = load_data(aug_dir + 'data' + str(i) + '.h5')
    data_aug['x'], data_aug['y'] = x, y
    y_pred = model_aug[i].predict(x)
    y_pred, y = utils.postprocessing(y_pred, y, y_min, y_max)
    accuracy_aug.append(utils.metric(utils.create_df(y, y_pred)))

print('fcnn')
print(get_mean_result(accuracy_fcnn))

x, y = load_data(fcnn_dir + 'data' + str(0) + '.h5')
data_fcnn['x'], data_fcnn['y'] = x, y
y_pred = model_fcnn[0].predict(x)
y_pred, y = utils.postprocessing(y_pred, y, y_min, y_max)

plot_corr_model(y, y_pred, 'plot_corr_fcnn.png')

print('pca')
print(get_mean_result(accuracy_pca))

x, y = load_data(pca_dir + 'data' + str(0) + '.h5')
data_pca['x'], data_pca['y'] = x, y
y_pred = model_pca[0].predict(x)
y_pred, y = utils.postprocessing(y_pred, y, y_min, y_max)

plot_corr_model(y, y_pred, 'plot_corr_pca.png')

print('augmentation')
print(get_mean_result(accuracy_aug))

x, y = load_data(aug_dir + 'data' + str(0) + '.h5')
data_aug['x'], data_aug['y'] = x, y
y_pred = model_aug[0].predict(x)
y_pred, y = utils.postprocessing(y_pred, y, y_min, y_max)

plot_corr_model(y, y_pred, 'plot_corr_aug.png')

metric_fcnn = []
metric_pca = []
metric_aug = []

critarion = 'MRE'

for i in range(k):
    metric_fcnn.append(accuracy_fcnn[i]['average'][critarion])
    metric_pca.append(accuracy_pca[i]['average'][critarion])
    metric_aug.append(accuracy_aug[i]['average'][critarion])

mae_fcnn = np.array(metric_fcnn).reshape(-1, 1)
mae_pca = np.array(metric_pca).reshape(-1, 1)
mae_aug = np.array(metric_aug).reshape(-1, 1)

data = np.concatenate([mae_fcnn, mae_pca, mae_aug], axis=1)
df_accuracy = pd.DataFrame(data, columns=['FCNN', 'PCA', "augmentation"])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

sns.boxplot(data=df_accuracy, ax=ax)
ax.set_ylabel('Mean relative error', fontsize=14)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)

plt.tight_layout()
plt.savefig('../picture/boxplot.png')

model_noise_fcnn = []
model_noise_pca = []
model_noise_aug = []

data_noise_fcnn = {}
data_noise_pca = {}
data_noise_aug = {}

accuracy_noise_fcnn = []
accuracy_noise_pca = []
accuracy_noise_aug = []

noise_fcnn_dir = '../result/noise_fcnn/'
noise_pca_dir = '../result/noise_pca/'
noise_aug_dir = '../result/noise_aug/'

k = 10

y_min = np.array([0, 0, 0.08])
y_max = np.array([180, 75, 0.78])

for i in range(k):
    model_noise_fcnn.append(load_model(noise_fcnn_dir + 'model' + str(i) +'.h5'))
    model_noise_pca.append(load_model(noise_pca_dir + 'model' + str(i) +'.h5'))
    model_noise_aug.append(load_model(noise_aug_dir + 'model' + str(i) +'.h5'))

    x, y = load_data(noise_fcnn_dir + 'data' + str(i) + '.h5')
    data_noise_fcnn['x'], data_noise_fcnn['y'] = x, y
    y_pred = model_noise_fcnn[i].predict(x)
    y_pred, y = utils.postprocessing(y_pred, y, y_min, y_max)
    accuracy_noise_fcnn.append(utils.metric(utils.create_df(y, y_pred)))

    x, y = load_data(noise_pca_dir + 'data' + str(i) + '.h5')
    data_noise_pca['x'], data_noise_pca['y'] = x, y
    y_pred = model_noise_pca[i].predict(x)
    y_pred, y = utils.postprocessing(y_pred, y, y_min, y_max)
    accuracy_noise_pca.append(utils.metric(utils.create_df(y, y_pred)))

    x, y = load_data(noise_aug_dir + 'data' + str(i) + '.h5')
    data_noise_aug['x'], data_noise_aug['y'] = x, y
    y_pred = model_noise_aug[i].predict(x)
    y_pred, y = utils.postprocessing(y_pred, y, y_min, y_max)
    accuracy_noise_aug.append(utils.metric(utils.create_df(y, y_pred)))

print('noise fcnn')
print(get_mean_result(accuracy_noise_fcnn))

x, y = load_data(noise_fcnn_dir + 'data' + str(0) + '.h5')
data_noise_fcnn['x'], data_noise_fcnn['y'] = x, y
y_pred = model_noise_fcnn[0].predict(x)
y_pred, y = utils.postprocessing(y_pred, y, y_min, y_max)

plot_corr_model(y, y_pred, 'plot_corr_noise_fcnn.png')

print('noise pca')
print(get_mean_result(accuracy_noise_pca))

x, y = load_data(noise_pca_dir + 'data' + str(0) + '.h5')
data_noise_pca['x'], data_noise_pca['y'] = x, y
y_pred = model_noise_pca[0].predict(x)
y_pred, y = utils.postprocessing(y_pred, y, y_min, y_max)

plot_corr_model(y, y_pred, 'plot_corr_noise_pca.png')

print('noise augmentation')
print(get_mean_result(accuracy_noise_aug))

x, y = load_data(noise_aug_dir + 'data' + str(0) + '.h5')
data_noise_aug['x'], data_noise_aug['y'] = x, y
y_pred = model_noise_aug[0].predict(x)
y_pred, y = utils.postprocessing(y_pred, y, y_min, y_max)

plot_corr_model(y, y_pred, 'plot_corr_noise_aug.png')

metric_noise_fcnn = []
metric_noise_pca = []
metric_noise_aug = []

critarion = 'MRE'

for i in range(k):
    metric_noise_fcnn.append(accuracy_noise_fcnn[i]['average'][critarion])
    metric_noise_pca.append(accuracy_noise_pca[i]['average'][critarion])
    metric_noise_aug.append(accuracy_noise_aug[i]['average'][critarion])

mae_noise_fcnn = np.array(metric_noise_fcnn).reshape(-1, 1)
mae_noise_pca = np.array(metric_noise_pca).reshape(-1, 1)
mae_noise_aug = np.array(metric_noise_aug).reshape(-1, 1)

data = np.concatenate([mae_noise_fcnn, mae_noise_pca, mae_noise_aug], axis=1)
df_accuracy = pd.DataFrame(data, columns=['FCNN', 'PCA', "augmentation"])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

sns.boxplot(data=df_accuracy, ax=ax)
ax.set_ylabel('Mean relative error', fontsize=14)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)

plt.tight_layout()
plt.savefig('../picture/boxplot_noise.png')

def split_data_for_noise_test(x_data, y_data, random_state):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
                                        x_data, y_data,
                                        test_size=0.15,
                                        random_state=random_state)

    return x_train, y_train,  x_test, y_test

hdf = '../data/spectrum.h5'

file =  h5py.File(hdf, 'r')

spectra = np.array(file['spectrum'])
phi = np.array(file['phi'])
theta = np.array(file['theta'])
lp = np.array(file['lp'])

target = np.concatenate([phi.reshape(-1, 1),
                         theta.reshape(-1, 1),
                         lp.reshape(-1, 1)], axis=1)

noise_estimation = utils.get_std(spectra.copy(), target.copy())
noise_accuracy = []

seed = [628, 693, 847, 621, 861, 409, 74, 306, 884, 777]
for i in range(k):
    spectra_noise = utils.add_noise(spectra.copy(), noise_estimation.copy())
    x_train, y_train, x_test, y_test = split_data_for_noise_test(spectra_noise.copy(), target.copy(), seed[i])

    x_train, x_test, _, _ = utils.preprocessing(x_train, x_test)
    y_train, y_test, _, _ = utils.preprocessing(y_train, y_test)

    y_pred = model_fcnn[i].predict(x_test)

    y_pred, y_test = utils.postprocessing(y_pred, y_test, y_min, y_max)

    noise_accuracy.append(utils.metric(utils.create_df(y_test, y_pred)))

    if i == 0:
        plot_corr_model(y_test, y_pred, 'corr_noise.png')

print('model trained on clean data and tested on noisy data')
print(get_mean_result(noise_accuracy))