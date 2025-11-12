import numpy as np
from numba import int64, float64, njit
from numba.experimental import jitclass
from scipy.interpolate import interp1d

from module.constant import data_config

BEAM_CONSTANT = 300e-7 * 2.28 / 13.1
SCATTERING_FACTOR = 13.6
LOG_COEFFICIENT = 0.038

spec_beams = [
    ('angle_x', float64),
    ('angle_y', float64),
    ('spectrum', int64[:]),
    ('angle_dist', float64[:]),
    ('Energy', float64[:]),

    ('ANGLE_COEFFICIENT', float64),
    ('BASELINE_ANGLE', float64),
]

spec_3db = [
    ('samples', float64[:, :]),
    ('x', float64[:]),
    ('y', float64[:]),
    ('z', float64[:]),

    ('ANGLE_COEFFICIENT', float64),
    ('BASELINE_ANGLE', float64),
]


@jitclass(spec_beams)
class GeneratorBeams:
    def __init__(self, angle_x, angle_y, spectrum, angle_dist, Energy, ANGLE_COEFFICIENT, BASELINE_ANGLE):
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_dist = angle_dist
        self.Energy = Energy
        self.spectrum = spectrum

        self.ANGLE_COEFFICIENT = ANGLE_COEFFICIENT
        self.BASELINE_ANGLE = BASELINE_ANGLE

    def iterate(self):
        for i in range(len(self.Energy)):
            energy = self.Energy[i]
            angle_std = self._calculate_angle_std(energy)
            
            for _ in range(self.spectrum[i]):
                alpha = np.random.normal(loc=self.angle_x, scale=angle_std)
                beta = np.random.normal(loc=self.angle_y, scale=angle_std)
                scat_angle = self._calculate_scattering_angle(energy)
                yield scat_angle, energy, alpha, beta

    def _calculate_angle_std(self, energy):
        return (self.angle_dist[0] / (1 + (energy / self.angle_dist[1]) ** 2)
                + self.angle_dist[2])

    def _calculate_scattering_angle(self, energy):
        return (SCATTERING_FACTOR / energy * np.sqrt(BEAM_CONSTANT) *
                (1 + LOG_COEFFICIENT * np.log(BEAM_CONSTANT)) * self.ANGLE_COEFFICIENT +
                self.BASELINE_ANGLE)


def resize_pgen(y, Energy):
    f = interp1d(np.linspace(Energy.min(), Energy.max(), y.shape[0]), y)
    return f(np.linspace(Energy.min(), Energy.max(), data_config['shapes'][0]))

@njit
def distribution_generation(spectrum, angle_x, angle_y, angle_dist, Energy, rangex, rangey, shapes):
    distribution = np.zeros(shapes)
    
    for i in range(Energy.shape[0]):
        energy = Energy[i]
        angle_std = angle_dist[0] / (1 + (energy / angle_dist[1]) ** 2) + angle_dist[2]
        
        for j, ax in enumerate(np.linspace(rangex[0], rangex[1], distribution.shape[1])):
            for k, ay in enumerate(np.linspace(rangey[0], rangey[1], distribution.shape[2])):
                exponent = (-(angle_x - ax)**2 - (angle_y - ay)**2) / (2 * angle_std**2)
                distribution[i, j, k] = np.exp(exponent)


    distribution = spectrum[:, None, None] * (distribution /  distribution.sum(axis=1).sum(axis=1)[:, None, None])
    return distribution

def get_samples(distribution, N):
    edges = [
        np.linspace(data_config['energy'].min(), data_config['energy'].max(), distribution.shape[0]+1),
        np.linspace(data_config['rangex'][0], data_config['rangex'][1], distribution.shape[1]+1),
        np.linspace(data_config['rangey'][0], data_config['rangey'][1], distribution.shape[2]+1)
    ]

    steps = [
        edges[0][1] - edges[0][0],
        edges[1][1] - edges[1][0],
        edges[2][1] - edges[2][0]
    ]

    if np.any(distribution):
        flat_dist = (distribution / np.sum(distribution)).flatten()
        distribution[np.isnan(distribution)] = 0
    else:
        flat_dist = None
    
    indices = np.random.choice(np.prod(distribution.shape), size=N, p=flat_dist)
    
    i, j, k = np.unravel_index(indices, distribution.shape)
    
    samples = np.column_stack([
        np.random.uniform(edges[0][i], edges[0][i+1]) - steps[0] / 2,
        np.random.uniform(edges[1][j], edges[1][j+1]) - steps[1] / 2,
        np.random.uniform(edges[2][k], edges[2][k+1]) - steps[2] / 2,
    ])
    
    return samples

@jitclass(spec_3db)
class Generator3DBeam:
    def __init__(self, samples,ANGLE_COEFFICIENT, BASELINE_ANGLE):
        self.samples = samples

        self.ANGLE_COEFFICIENT = ANGLE_COEFFICIENT
        self.BASELINE_ANGLE = BASELINE_ANGLE

    def iterate(self):
        for energy, alpha, beta in self.samples:
            scat_angle = self._calculate_scattering_angle(energy)
            yield scat_angle, energy, alpha, beta

    def _calculate_scattering_angle(self, energy):
        return (SCATTERING_FACTOR / energy * np.sqrt(BEAM_CONSTANT) *
                (1 + LOG_COEFFICIENT * np.log(BEAM_CONSTANT)) * self.ANGLE_COEFFICIENT +
                self.BASELINE_ANGLE)