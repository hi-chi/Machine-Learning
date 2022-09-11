from pylab import *
from datetime import datetime
import h5py
from numpy import random

from joblib import Parallel, delayed

import utils

Ndata = 50000  # number of cases to generate
directory = '../Flux_cop/'  # where the basic cases are stored
savedir = '../data_split/'  # where to save the generated cases
sizey = 150  # transerse size of the intensity profiles
nwavelength = 3  # Number of wavelengths
npos = 1  # number of intensity profiles around focus
nstep_flux = 60  # number of steps for the flux computation
npos_wlength = 5  # number of values for the wavelengths w1 and w2

tmin = 0
tmax = 0.6
anglemin = 0
anglemax = 2 * pi
intmin = 0.5
intmax = 2.0
nval_tilt = 16
nval_angle = 41
nval_int = 11
tstep = (tmax - tmin) / (nval_tilt - 1)
anglestep = (anglemax - anglemin) / (nval_angle - 1)
intstep = (intmax - intmin) / (nval_int - 1)

wavelength0 = 2 * pi * 3e8 / 800e-9

run_param = {'tmin': tmin, 'tmax': tmax,
             'anglemin': anglemin, 'anglemax': anglemax,
             'intmin': intmin, 'intmax': intmax,
             'nval_tilt': nval_tilt, 'nval_angle': nval_angle, 'nval_int': nval_int,
             'tstep': tstep, 'anglestep': anglestep, 'intstep': intstep}

print(run_param)

def rotation(angle, matfield, indmid):
    val = zeros([npos, nstep_flux, 2, matfield[0, 0, 0, 0, :].size, matfield[0, 0, 0, 0, :].size])
    for ii in range(matfield[0, 0, 0, 0, :].size):
        for jj in range(matfield[0, 0, 0, 0, :].size):
            ind0 = cos(angle) * (ii - indmid) - sin(angle) * (jj - indmid) + indmid
            ind1 = sin(angle) * (ii - indmid) + cos(angle) * (jj - indmid) + indmid
            # interpolation
            if (ind0 > 0 and ind0 < matfield[0, 0, 0, 0, :].size - 1 and 0 < ind1 < matfield[0, 0, 0, 0,
                                                                                    :].size - 1):
                val[:, :, :, ii, jj] = matfield[:, :, :, int(ind0), int(ind1)] * (int(ind0) + 1 - ind0) * (
                        int(ind1) + 1 - ind1) + matfield[:, :, :, int(ind0) + 1, int(ind1)] * (ind0 - int(ind0)) * (
                                               int(ind1) + 1 - ind1) + matfield[:, :, :, int(ind0),
                                                                       int(ind1) + 1] * (int(ind0) + 1 - ind0) * (
                                               ind1 - int(ind1)) + matfield[:, :, :, int(ind0), int(ind1)] * (
                                               ind0 - int(ind0)) * (ind1 - int(ind1))
    return val


def run_(icase, param):
    iwave1 = param[0][0]
    iwave2 = param[0][1]

    intensity, angle, tilt = param[1], param[2], param[3]

    output_file = savedir + '/case%06d' % icase
    Efield = zeros([npos, nstep_flux, 1, sizey, sizey])
    Bfield = zeros([npos, nstep_flux, 1, sizey, sizey])

    wavelength1 = 2 * pi * 3e8 / (600e-9 + 50e-9 * iwave1)
    wavelength2 = 2 * pi * 3e8 / (800e-9 + 50e-9 * iwave2)

    # for each wavelength
    for iwlgh in range(nwavelength):
        if iwlgh == 0:
            wavewrite = '/wavelength%1d' % iwlgh + 'pos%1d' % iwave1 + 'pos%1d' % iwave2
        elif iwlgh == 1:
            wavewrite = '/wavelength%1d' % iwlgh + 'pos%1d' % iwave1
        elif iwlgh == 2:
            wavewrite = '/wavelength%1d' % iwlgh + 'pos%1d' % iwave2

        Eread = zeros([npos, nstep_flux, 2, sizey, sizey])
        Bread = zeros([npos, nstep_flux, 2, sizey, sizey])
        # for each position around focus

        for jj in range(npos):
            for ii in range(nstep_flux):
                ffile = directory + wavewrite + '/PFT0.%04d' % (
                        tilt[iwlgh] * 10000) + '/Fieldpos%1d' % jj + '/Efield%03d' % ii
                file = h5py.File(ffile + '.h5', 'r')
                Eread[jj, ii, :, :, :] = np.array(file['x_data'][:, :, :])
                file.close()
                ffile = directory + wavewrite + '/PFT0.%04d' % (
                        tilt[iwlgh] * 10000) + '/Fieldpos%1d' % jj + '/Bfield%03d' % ii
                file = h5py.File(ffile + '.h5', 'r')
                Bread[jj, ii, :, :, :] = np.array(file['x_data'][:, :, :])
                file.close()
        Eread = rotation(angle[iwlgh], Eread, (sizey - 1) / 2)
        Bread = rotation(angle[iwlgh], Bread, (sizey - 1) / 2)
        Efield = Efield + Eread * intensity[iwlgh]
        Bfield = Bfield + Bread * intensity[iwlgh]

    flux = zeros([npos, sizey, sizey])
    for jj in range(npos):
        for ii in range(nstep_flux):
            flux[jj, :, :] = flux[jj, :, :] + Efield[jj, ii, 0, :, :] * Bfield[jj, ii, 1, :, :] - Efield[jj, ii, 1, :,
                                                                                                  :] * Bfield[jj, ii, 0,
                                                                                                       :, :]

        # filter to reduce the aliasing from rotation
        fluxfft = np.fft.fft2(flux[jj, :, :])
        r, c = fluxfft.shape
        keepfrac = 0.2
        fluxfft[int(r * keepfrac):int(r * (1 - keepfrac))] = 0
        fluxfft[:, int(c * keepfrac):int(c * (1 - keepfrac))] = 0
        flux[jj, :, :] = np.fft.ifft2(fluxfft).real

    return flux[:, :, :], np.array([iwave1, iwave2, intensity[1], intensity[2], angle[0], angle[1], angle[2], tilt[0], tilt[1], tilt[2]])

def par_gen(par_run, n_data):
    flux = np.zeros(shape=(n_data, 150, 150))
    params = np.zeros(shape=(n_data, 10))

    results = Parallel(n_jobs=16)(delayed(run_)(i, par_run[i]) for i in range(n_data))

    for i in range(len(results)):
        flux[i], params[i] = results[i][0], results[i][1]

    return flux, params

def gen(n_data, gen_fun, k, files):
    par_run = []
    for i in range(n_data):
        par_run.append(gen_fun(run_param))

    size_block = n_data // k

    for i in range(k):
        flux, params = par_gen(par_run[i*size_block:(i+1)*size_block], size_block)

        flux = utils.resize(flux)
        parname = ['iwave1', 'iwave2', 'intensity1','intensity2', 'angle0', 'angle1', 'angle2', 'tilt0', 'tilt1', 'tilt2']
        utils.save(flux, params, parname, name=files[i])


def union(k, n_data, file_name, files):
    x_data = []
    y_data = []

    names = ['iwave1', 'iwave2', 'intensity1','intensity2', 'angle0', 'angle1', 'angle2', 'tilt0', 'tilt1', 'tilt2']

    for i, file in enumerate(files):
        data = h5py.File(file, 'r')
        
        x_data.append(np.array(data['flux']))
        y_t = np.zeros((x_data[i].shape[0], 10))
        for i, n in enumerate(names):
            y_t[:, i] = np.array(data[n])

        y_data.append(y_t)

    utils.save(np.concatenate(x_data), np.concatenate(y_data), names, file_name)

np.random.seed(42)

start_time = datetime.now()
print('pdata')
k = 10
file_name = 'pdata'
files = [f'../data/{file_name}{i}.h5' for i in range(k)]
gen(Ndata, utils.gen_point, k, files=files)
union(k, Ndata, file_name=f'../data/{file_name}.h5', files=files)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


np.random.seed(43)

start_time = datetime.now()
print('pdata_fixw')
k = 10
file_name = 'pdata_fixw'
files = [f'../data/{file_name}{i}.h5' for i in range(k)]
gen(Ndata, utils.gen_point_fixw, k, files=files)
union(k, Ndata, file_name=f'../data/{file_name}.h5', files=files)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


np.random.seed(44)

start_time = datetime.now()
print('pdata')
k = 10
file_name = 'pdata_fixA'
files = [f'../data/{file_name}{i}.h5' for i in range(k)]
gen(Ndata, utils.gen_point_fixA, k, files=files)
union(k, Ndata, file_name=f'../data/{file_name}.h5', files=files)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
