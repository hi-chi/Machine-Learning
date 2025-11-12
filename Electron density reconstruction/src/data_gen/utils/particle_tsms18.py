import numpy as np
from numba import float64
from numba.experimental import jitclass

spec = [
    ('mass', float64),
    ('charge', float64),
    ('c', float64),
    ('velocity', float64[:]),
    ('coord', float64[:]),
]

@jitclass(spec)
class Particle:
    def __init__(self, mass, charge, velocity, coord, c):
        self.mass = mass
        self.charge = charge
        self.c = c
        self.velocity = velocity
        self.coord = coord

    def velFromMevAngle(self, E, phi, theta):
        En = E * 1e6 * self.charge  # энергия в Дж
        absV = self.c * np.sqrt(1 - 1 / (En / self.mass / self.c ** 2 + 1) ** 2)
        self.velocity[0] = absV * np.cos(phi) * np.sin(theta)
        self.velocity[1] = absV * np.sin(phi) * np.sin(theta)
        self.velocity[2] = absV * np.cos(theta)
        return self.velocity

    def velFromMevAngle1(self, E=100, alf=0, bet=0):
        En = E * 1e6 * self.charge  # энергия в Дж
        absV = self.c * np.sqrt(1 - 1 / (En / self.mass / self.c ** 2 + 1) ** 2)

        self.velocity[2] = absV * np.sin(bet + np.pi / 2) * np.cos(alf)
        self.velocity[1] = absV * np.sin(bet + np.pi / 2) * np.sin(alf)
        self.velocity[0] = absV * np.cos(bet + np.pi / 2)
        return self.velocity, absV

    def push(self, B, r0, v0, dt=1e-11):
        v1 = v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2
        sq = np.sqrt(1 - v1 / self.c ** 2)

        qm = self.charge / self.mass * sq
        Ax = 0
        Ay = qm * (-(-v0[2] * B))
        Az = qm * (-v0[1] * B)
        A = np.array([Ax, Ay, Az])
        v0 = v0 + A * dt
        r0 = r0 + v0 * dt + A * dt ** 2 / 2

        v2 = (v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2)
        # speed correction - energy should not increase
        corr = np.sqrt(v2 / v1)
        v0 = v0 / corr
        return np.array([r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]], dtype=np.float64)

    @staticmethod
    def magnet(r, rc):
        arg = (0.715 * np.exp(-(np.sqrt((r[1] - rc[1]) ** 2 + (r[2] - rc[2]) ** 2)) ** 7 / 0.031 ** 7) - 0.025 -
               0.055 * np.exp(-(np.sqrt((r[1] - rc[1]) ** 2 + (r[2] - rc[2]) ** 2) - 0.045) ** 2 / 0.02 ** 2))
        return arg

    @staticmethod
    def magnet1(r, rc):
        arg = (0.45 * np.exp(-(np.sqrt((r[1] - rc[1]) ** 2 + (r[2] - rc[2]) ** 2)) ** 7 / 0.061 ** 7) - 0.025 -
               0.055 * np.exp(-(np.sqrt((r[1] - rc[1]) ** 2 + (r[2] - rc[2]) ** 2) - 0.045) ** 2 / 0.02 ** 2))
        return arg

    @staticmethod
    def magnet2(r, rc):
        arg = (0.715 * np.exp(-(np.sqrt((r[1] - rc[1]) ** 2 + (r[2] - rc[2]) ** 2)) ** 7 / 0.031 ** 7) - 0.025 -
               0.055 * np.exp(-(np.sqrt((r[1] - rc[1]) ** 2 + (r[2] - rc[2]) ** 2) - 0.045) ** 2 / 0.02 ** 2))
        return arg