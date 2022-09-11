import h5py
import numpy as np
from numpy import random
import cv2

def gen_point(run_param):
    iwave1 = np.random.randint(low=0, high=4)
    iwave2 = np.random.randint(low=1, high=5)

    intensity = [1, run_param['intmin'], run_param['intmin']] + \
                random.randint([1, run_param['nval_int'], run_param['nval_int']]) * \
                run_param['intstep']
    angle = run_param['anglemin'] + \
            random.randint([run_param['nval_angle'], run_param['nval_angle'], run_param['nval_angle']]) * \
            run_param['anglestep']
    tilt = run_param['tmin'] + \
            random.randint([run_param['nval_tilt'], run_param['nval_tilt'], run_param['nval_tilt']]) * \
            run_param['tstep']

    return [iwave1, iwave2], intensity, angle, tilt


def gen_point_fixw(run_param):
    iwave1 = np.random.randint(low=1, high=3)
    iwave2 = np.random.randint(low=2, high=4)

    intensity = [1, run_param['intmin'], run_param['intmin']] + \
                random.randint([1, run_param['nval_int'], run_param['nval_int']]) * \
                run_param['intstep']
    angle = run_param['anglemin'] + \
            random.randint([run_param['nval_angle'], run_param['nval_angle'], run_param['nval_angle']]) * \
            run_param['anglestep']
    tilt = run_param['tmin'] + \
            random.randint([run_param['nval_tilt'], run_param['nval_tilt'], run_param['nval_tilt']]) * \
            run_param['tstep']

    return [iwave1, iwave2], intensity, angle, tilt

def gen_point_fixA(run_param):
    iwave1 = np.random.randint(low=0, high=4)
    iwave2 = np.random.randint(low=1, high=5)

    intensity = [1, 0.95, 0.95] + random.randint([1, 4, 4]) * 0.15
    angle = run_param['anglemin'] + \
            random.randint([run_param['nval_angle'], run_param['nval_angle'], run_param['nval_angle']]) * \
            run_param['anglestep']
    tilt = run_param['tmin'] + \
            random.randint([run_param['nval_tilt'], run_param['nval_tilt'], run_param['nval_tilt']]) * \
            run_param['tstep']

    return [iwave1, iwave2], intensity, angle, tilt

def save(flux, params, parname, name='../data/data128.h5'):
    hf = h5py.File(name, 'w')

    hf.create_dataset('flux', data=flux)
    for i, n in enumerate(parname):
        hf.create_dataset(n, data=params[:, i])

    hf.close()

def resize(flux, dim=(50, 50)):
    flux_resize = np.zeros((flux.shape[0], dim[0], dim[1]))

    for i in range(flux.shape[0]):
        flux_resize[i] = cv2.resize(flux[i, 35:115, 35:115], dim)

    return flux_resize