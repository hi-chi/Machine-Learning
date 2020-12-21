"""Module for generating an electron beam."""

import pyHiChi as pfc

LIGHT_FS = 45


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
