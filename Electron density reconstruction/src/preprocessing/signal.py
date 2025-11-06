import numpy as np
from scipy.interpolate import interp1d


def resize(data, xmin, xmax, new_size):
    if len(data.shape) > 1:
        f = interp1d(np.linspace(xmin, xmax, data.shape[1]), data, 'linear', fill_value='extrapolate')
        return f(np.linspace(xmin, xmax, new_size))
    else:
        f = interp1d(np.linspace(xmin, xmax, data.shape[0]), data, 'linear', fill_value='extrapolate')
        return f(np.linspace(xmin, xmax, new_size))

