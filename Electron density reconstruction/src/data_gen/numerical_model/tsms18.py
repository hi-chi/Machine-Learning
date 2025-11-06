import time
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy import special

from module.constant import data_config
from src.data_gen.utils.save_dataset import save_year18
from src.data_gen.utils.particle_tsms18 import Particle
from src.data_gen.utils import geometry_tsms18

from src.data_gen.utils.pygenerators import GeneratorBeams, distribution_generation, get_samples, Generator3DBeam
from src.preprocessing import signal


# for precise calculation of particle entry into the magnetic field of a round magnet
# particle coordinate (y, z), k of a straight particle, magnet radius, magnet coordinate (y, z)
@njit
def sec(y_p, z_p, k_p, r_m, y_m, z_m):
    b_p = z_p - k_p * y_p  # коэф b уравнения прямой частицы
    a = (1 + 1 / k_p / k_p)
    b = -2 * (z_m + (b_p / k_p + y_m) / k_p)
    c = z_m * z_m + (b_p / k_p + y_m) * (b_p / k_p + y_m) - r_m * r_m
    D = b * b - 4 * a * c
    z = (-b - np.sqrt(D)) / 2 / a
    if np.isnan(z):
        z = z_m - r_m
    return z


# to calculate the intersection point with the screen if the screen is at an angle to the axis
# particle coordinate (y, z), k of the direct particle, screen angle, screen coordinate (y, z)
@njit
def sec_scr(y_p, z_p, k_p, alfa, y_sc, z_sc):
    b_p = z_p - k_p * y_p  # коэф b уравнения прямой частицы
    k_sc = np.tan(np.pi - alfa * np.pi / 180)  # коэф k прямой экрана
    b_sc = z_sc - k_sc * y_sc  # коэф b уравнения прямой экрана
    z = (-k_sc / k_p * b_p + b_sc) / (1 - k_sc / k_p)
    return z


def gauss(x, arg):
    return arg[0]*np.exp(-(x-arg[1])**2/arg[2]**2)


def beam(Erange, sig_x, sig_y):
    angle_x = sig_x
    angle_y = sig_y

    spec = np.zeros((len(Erange),))

    spec = spec + gauss(Erange, [random.random(), 0., 10 + 50 * random.random()])

    ng = random.randint(1, 6)
    es0 = []
    des = []
    maxE0 = 0
    dEmax0 = 0

    random_value = random.choice([50, 100, 500, 1200])
    for j in range(0, ng):
        e0 = random.random() * random_value
        de = e0 / random.randint(1, 10)

        if e0 > maxE0:
            maxE0 = e0 * 1
            dEmax0 = de * 1
        es0.append(e0)
        des.append(de)

    for j in range(0, ng):
        a = 10 * random.random()
        if es0[j] == maxE0:
            a = 1

        spec = spec + gauss(Erange, [a, es0[j], des[j]])
    if maxE0 > 50:
        spec = spec * (1 - special.erf((Erange - maxE0) / dEmax0 * 10)) / 2  # тут делаем отсчеку

    ang = [
        0.002 + 0.006 * random.random(),
        100.0 + 100.0 * random.random(),
        0.4e-3 * (1 + 1 * random.random()),
    ]

    return angle_x, angle_y, spec, np.float64(ang)


def gen_beams():
    a_x = 0.1 * (random.random() * (0.25 + 0.15) - 0.15)
    a_y = 0

    angle_x, angle_y, spectrum, angle_dist = beam(data_config['energy'], sig_x=a_x, sig_y=a_y)

    return angle_x, angle_y, spectrum, angle_dist

@njit
def simulation(pgen):
    screen1 = np.zeros((geometry_tsms18.ny1, geometry_tsms18.nx1))
    screen2 = np.zeros((geometry_tsms18.ny2, geometry_tsms18.nx2))

    dt0 = 1e-11

    part = Particle(9.1e-31, 1.6e-19, np.zeros(3), np.zeros(3), 2.9998e8)
    for scat_angle, energy, alpha, beta in pgen.iterate():
        v0, absV0 = part.velFromMevAngle1(energy, alpha, beta)
        r0 = np.array([0., 0., 0.], dtype=np.float64)
        v0 = np.array([v0[0], v0[1], v0[2]], dtype=np.float64)

        # time step correction
        R = np.sqrt((absV0 ** 2 - v0[1] ** 2) / (1 - (absV0 ** 2) / part.c ** 2)) * part.mass / part.charge / 0.7
        dt = dt0 * np.log(np.sqrt(R) / 0.03)

        # distance to 1 magnet taking into account the angle
        t1 = (sec(0, 0, v0[2] / v0[1], geometry_tsms18.rmag, 0, geometry_tsms18.l1)) / v0[2]
        r0 += v0 * t1

        # in 1 magnet
        while np.sqrt((r0[2] - geometry_tsms18.l1) ** 2 + r0[1] ** 2) < (geometry_tsms18.rmag + 0.001):
            res = part.push(part.magnet1(r0, [0., 0., geometry_tsms18.l1]), r0, v0, dt)
            r0 = np.ascontiguousarray(res[:3])
            v0 = np.ascontiguousarray(res[3:])

        # to the first screen
        t2 = (sec_scr(r0[1], r0[2], v0[2] / v0[1], 1, 0, geometry_tsms18.l3) - (r0[2])) / v0[2]
        r0 = r0 + v0 * t2

        yi = int((r0[1] - geometry_tsms18.x_min1) / geometry_tsms18.dx1)
        xi = int((r0[0] - geometry_tsms18.y_min1) / geometry_tsms18.dx1)

        seed = 4
        if seed <= xi < geometry_tsms18.ny1 - seed and seed <= yi < geometry_tsms18.nx1 - seed:
            for i_n in range(-seed, seed):
                for j_n in range(-seed, seed):
                    screen1[xi + i_n, yi + j_n] += np.exp(-(i_n ** 2 + j_n ** 2) / seed ** 2)

        #### small-angle scattering:
        # - three angles (the sum of the squares must be equal to the square of the scattering angle),
        # their values are random, but do not exceed the scattering angle
        # - the scattering angle depends strictly on the energy, materials, and thickness of the screen layers
        # - rotation by rotation matrices

        scat_a = scat_angle * random.random() * np.sign(random.normalvariate(mu=0.0, sigma=1))
        scat_b = np.sqrt(scat_angle ** 2 - scat_a ** 2) * random.random() * np.sign(
            random.normalvariate(mu=0.0, sigma=1)
        )
        scat_g = np.sqrt(scat_angle ** 2 - scat_b ** 2 - scat_a ** 2) * np.sign(
            random.normalvariate(mu=0.0, sigma=1)
        )

        Mx = np.array([[1., 0., 0.],
                       [0., np.cos(scat_a), -np.sin(scat_a)],
                       [0., np.sin(scat_a), np.cos(scat_a)]], dtype=np.float64)
        My = np.array([[np.cos(scat_b), 0., np.sin(scat_b)],
                       [0., 1., 0.],
                       [-np.sin(scat_b), 0., np.cos(scat_b)]], dtype=np.float64)
        Mz = np.array([[np.cos(scat_g), -np.sin(scat_g), 0.],
                       [np.sin(scat_g), np.cos(scat_g), 0.],
                       [0., 0., 1.]], dtype=np.float64)
        v0 = np.dot(v0, Mx)
        v0 = np.dot(v0, My)
        v0 = np.dot(v0, Mz)
        v0[np.isnan(v0) == True] = 0

        # distance to 2 magnets taking into account the angle
        t3 = (sec(r0[1], r0[2], v0[2] / v0[1], geometry_tsms18.rmag, 0, geometry_tsms18.l2) - r0[2]) / v0[2]
        r0 += v0 * t3

        # in 2 magnets
        while np.sqrt((r0[2] - geometry_tsms18.l2) ** 2 + r0[1] ** 2) < (geometry_tsms18.rmag + 0.001):
            res = part.push(part.magnet2(r0, [0., 0., geometry_tsms18.l2]), r0, v0, dt)
            r0 = np.ascontiguousarray(res[:3])
            v0 = np.ascontiguousarray(res[3:])

        # to the second screen
        t4 = (sec_scr(r0[1], r0[2], v0[2] / v0[1], 1, 0, geometry_tsms18.l4) - (r0[2])) / v0[2]
        r0 += v0 * t4

        yi = int((r0[1] - geometry_tsms18.x_min2) / geometry_tsms18.dx2)
        xi = int((r0[0] - geometry_tsms18.y_min2) / geometry_tsms18.dx2)

        seed = 4
        if seed <= xi < geometry_tsms18.ny2 - seed and seed <= yi < geometry_tsms18.nx2 - seed:
            for i_n in range(-seed, seed):
                for j_n in range(-seed, seed):
                    screen2[xi + i_n, yi + j_n] += np.exp(-(i_n ** 2 + j_n ** 2) / seed ** 2)

    return screen1, screen2



def generate_pg__(mode, N):
    angle_x, angle_y, spectrum, angle_dist = gen_beams()

    if mode == 0:
        pgen = GeneratorBeams(
            angle_x,  angle_y,
            np.array(spectrum * N / spectrum.sum(), np.int64),
            angle_dist, data_config['energy'], data_config['ANGLE_COEFFICIENT'],  data_config['BASELINE_ANGLE']
        )
    elif mode == 1:
        resized_spectrum = signal.resize(spectrum, data_config['energy'].min(), data_config['energy'].max(), data_config['shapes'][0])
        resized_energy = np.linspace(data_config['energy'].min(), data_config['energy'].max(), resized_spectrum.shape[0])

        distribution = distribution_generation(
            resized_spectrum, angle_x, angle_y, angle_dist, resized_energy,
            rangex=data_config['rangex'], rangey=data_config['rangey'], shapes=(256, 256, 256)
        )
        samples = get_samples(distribution, N)
        pgen = Generator3DBeam(samples, data_config['ANGLE_COEFFICIENT'],  data_config['BASELINE_ANGLE'])
    else:
        raise ValueError("Mode must be 0 or 1")

    screen1, screen2 = simulation(pgen)

    return screen1, screen2, angle_x, angle_y, spectrum, angle_dist


def generate(idx, save_args, mode, N):
    print(f'process id: {idx}')
    i1, i2, angle_x, angle_y, spectrum, angle_dist = generate_pg__(mode=mode, N=N)

    path = save_args['directory'] + f'{idx}'
    save_year18(path, save_args, i1, i2, angle_x, angle_y, spectrum, angle_dist)


if __name__ == "__main__":
    data_config.set_global_key('18_base')
    for i in range(2):
        mode = i

        np.random.seed(42)
        random.seed(42)

        start_time = time.time()

        screen1, screen2, angle_x, angle_y, spectrum, angle_dist = generate_pg__(mode, N=100_000)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(elapsed_time)

        screen1 = cv2.resize(screen1, (384, 128))
        screen2 = cv2.resize(screen2, (384, 128))

        np.savez('file', screen1 = screen1)

        fig, ax = plt.subplots(2, 2, figsize=(13, 6))

        im1 = ax[0, 0].imshow(screen1, cmap='jet')
        fig.colorbar(im1, ax=ax[0, 0], orientation='vertical', shrink=0.4, aspect=20)

        im2 = ax[0, 1].imshow(screen2, cmap='jet')
        fig.colorbar(im2, ax=ax[0, 1], orientation='vertical', shrink=0.4, aspect=20)

        ax[1, 0].plot(data_config['energy'], spectrum)
        v = angle_dist[0] / (1 + (data_config['energy'] / angle_dist[1]) ** 2) + angle_dist[2]
        ax[1, 1].plot(data_config['energy'], v * 1e3)
        ax[1, 1].set_ylim(0, 5 * 10)

        plt.tight_layout()
        plt.show()
        plt.savefig(f'image_low_resolution_q{mode}.png')
        