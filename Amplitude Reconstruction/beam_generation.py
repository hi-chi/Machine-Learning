"""Module for setting fields and generating an electron beam."""

import numpy as np
from numba import cfunc
import pyHiChi as pfc

LIGHT_FS = 45
a0 = None


def gen_electron(gamma):
    """Electron creation.

    Args:
        gamma: energy

    Returns:
        electron
    """
    pos = pfc.Vector3d(0, 0, 0)

    momentum_x = gamma * pfc.LIGHT_VELOCITY * pfc.ELECTRON_MASS
    momentum = pfc.Vector3d(momentum_x, 0, 0)
    particle = pfc.Particle(pos, momentum, 1.0, pfc.ELECTRON)

    return particle


def gen_beam(number_electrons, gamma):
    """Electron beam creation.

    Args:
        number_electrons: the number of electrons for the experiment
        gamma: energy

    Returns:
        electron beam
    """
    beam = pfc.Ensemble()

    for _ in range(number_electrons):
        particle = gen_electron(gamma)
        beam.add(particle)

    return beam


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
    else:
        return 0


def Bz(x, y, z, t):
    c = pfc.LIGHT_VELOCITY
    lymda = 10 ** -4
    fx = pfc.pi * (x + c * t)
    if 0 <= x < lymda:
        e_rel = (2 * pfc.pi * pfc.ELECTRON_MASS * c ** 2) / (lymda * np.abs(pfc.ELECTRON_CHARGE))
        return - (a0 * e_rel * (np.sin(fx / lymda)) ** 2 * np.sin((2 * fx) / lymda))
    else:
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
