import numpy as np
import pandas as pd
import h5py
import itertools

from sklearn.metrics import r2_score

pnt = ['phi', 'theta', 'Lp']
pnp = ['phi_pred', 'theta_pred', 'Lp_pred']

def create_df(y_true, y_pred):
    df = np.concatenate([y_true, y_pred], axis=1)

    df = pd.DataFrame(df, columns=['phi', 'theta', 'Lp', 'phi_pred', 'theta_pred', 'Lp_pred'])

    df['diff_phi'] = df['phi'] - df['phi_pred']
    df['diff_theta'] = df['theta'] - df['theta_pred']
    df['diff_lp'] = df['Lp'] - df['Lp_pred']

    return df

def relitive_(true_, pred_):
    sum_ = 0
    for i in range(true_.shape[0]):
        sum_ += np.abs(true_[i] - pred_[i]) / (np.max(true_) * true_.shape[0])
    return sum_ * 100

def metric(df):
    k_round = 3

    mre_ = []
    r2_ = []

    for i in range(len(pnt)):
        r2_.append(np.around(r2_score(df[pnt[i]], df[pnp[i]]), k_round))
        mre_.append(np.around(relitive_(df[pnt[i]], df[pnp[i]]), k_round))

    result_metric = np.array([mre_ + [np.around(np.mean(mre_), k_round), ],
                              r2_ + [np.around(np.mean(r2_), k_round), ],
            ])

    index_ = ["MRE", "R2"]
    columns_ = ["phi", "theta", "Lp", "average"]

    return pd.DataFrame(result_metric, columns=columns_, index=index_)

def preprocessing(train, test):
    min_ = np.zeros(train.shape[1])
    max_ = np.zeros(train.shape[1])

    for i in range(min_.shape[0]):
        min_[i] = np.min(train[:,i])
        max_[i] = np.max(train[:,i])

    for i in range(min_.shape[0]):
        if min_[i] == max_[i]:
            train[:,i] = 0.0
            test[:,i] = 0.0
        else:
            train[:,i] = 2.0 * (train[:,i] - min_[i]) / (max_[i] - min_[i]) - 1.0
            test[:,i] = 2.0 * (test[:,i] - min_[i]) / (max_[i] - min_[i]) - 1.0

    return train, test, min_, max_

def postprocessing(test, pred, min_, max_):
    for i in range(test.shape[1]):
        test[:,i] = min_[i] + (max_[i] - min_[i]) * (test[:,i] + 1.0) * 0.5
        pred[:,i] = min_[i] + (max_[i] - min_[i]) * (pred[:,i] + 1.0) * 0.5

    return test, pred

def save_data(x_test, y_test, i, path):
    output_file = path + 'data' + str(i) + '.h5'
    hf = h5py.File(output_file, 'w')
    hf.create_dataset('x_test', data=x_test)
    hf.create_dataset('y_test', data=y_test)

    hf.close()

def step_calc(data):
    list_ = list(set(data))
    list_.sort()

    step = list_[1] - list_[0]
    return step

def get_std(data, target):
    nearest = {}

    step_phi = [step_calc(target[:, 0]), 0, -step_calc(target[:, 0])]
    step_theta = [step_calc(target[:, 1]), 0, -step_calc(target[:, 1])]
    step_lp = [step_calc(target[:, 2]), 0, -step_calc(target[:, 2])]

    for i in range(data.shape[0]):
        ne = []

        for p, t, l in itertools.product(step_phi, step_theta, step_lp):
            key = (np.isclose(target[:, 0], target[i, 0] + p, rtol=1e-05) &
                  np.isclose(target[:, 1], target[i, 1] + t, rtol=1e-05) &
                  np.isclose(target[:, 2], target[i, 2] + l, rtol=1e-05))
            if key.sum() != 0:
                ne.append(data[key])
        ne = np.array(ne)
        nearest[i] = np.std(ne, axis=0)[0]

    return  np.array(list(nearest.values()))

def add_noise(data, sigma, coef=1):
    noise = data.copy()
    noise += np.random.normal(0, sigma*coef, size=noise.shape)

    return noise
