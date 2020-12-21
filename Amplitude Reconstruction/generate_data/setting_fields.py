"""Module for setting fields."""

import numpy as np
from numba import cfunc
import pyHiChi as pfc

a0 = None


def to_cfunc_fieldt(field_func):
    cfield_func = cfunc("float64(float64,float64,float64,float64)", nopython=True)(field_func)
    return cfield_func.address


def Ey(x, y, z, t):
    c = pfc.LIGHT_VELOCITY
    lymda = 10 ** -4
    fx = pfc.pi * (x + c * t)
    if 0 <= x < lymda:
        e_rel = (2 * pfc.pi * pfc.ELECTRON_MASS * c ** 2) / (lymda * np.abs(pfc.ELECTRON_CHARGE))
        return a0 * e_rel * (np.sin(fx / lymda)) ** 2 * np.sin((2 * fx) / lymda)

    return 0


def Bz(x, y, z, t):
    c = pfc.LIGHT_VELOCITY
    lymda = 10 ** -4
    fx = pfc.pi * (x + c * t)
    if 0 <= x < lymda:
        e_rel = (2 * pfc.pi * pfc.ELECTRON_MASS * c ** 2) / (lymda * np.abs(pfc.ELECTRON_CHARGE))
        return - (a0 * e_rel * (np.sin(fx / lymda)) ** 2 * np.sin((2 * fx) / lymda))

    return 0


def field0(x, y, z, t):
    return 0


def gen_func():
    """Compiling field functions.

    Returns:
        fields
    """

    c0 = to_cfunc_fieldt(field0)
    cBz = to_cfunc_fieldt(Bz)
    cEy = to_cfunc_fieldt(Ey)
    return (c0, cEy, c0, c0, c0, cBz)


def step(min_coords, max_coords, grid_size):
    """Calculating grid steps.

    Args:
        min_coords: minimum coordinate of the computational domain
        max_coords: maximum coordinate of the computational domain
        grid_size: computational domain size

    Returns:
        calculated step size
    """

    steps = pfc.Vector3d(1, 1, 1)

    steps.x = (max_coords.x - min_coords.x) / (grid_size.x)
    steps.y = (max_coords.y - min_coords.y) / (grid_size.y)
    steps.z = (max_coords.z - min_coords.z) / (grid_size.z)

    return steps
