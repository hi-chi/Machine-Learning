import numpy as np
import torch

from module.constant import device, data_config
from module_torch.utils import non_zero_norm
from src.train_electron_density_reconstruction.scripts.fourier_distributions_transform import transform_distribution


class DatasetCreate3DAdditive:
    def __init__(
        self,
        i1_data,
        i2_data,
        spectrum,
        ax,
        ay,
        ad,
        batch_size=32,
        device=device,
        max_iter_per_epoch=None,
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_samples_per_example = 10
        self.max_iter_per_epoch = max_iter_per_epoch

        self.i1_data = torch.tensor(i1_data).requires_grad_(False)
        self.i2_data = torch.tensor(i2_data).requires_grad_(False)
        self.spectrum = torch.tensor(spectrum).requires_grad_(False)
        self.ax = torch.tensor(ax).requires_grad_(False)
        self.ay = torch.tensor(ay).requires_grad_(False)
        self.ad = torch.tensor(ad).requires_grad_(False)

        self.shapes = data_config['shapes']
        energy = np.linspace(data_config['energy'].min(), data_config['energy'].max(), self.shapes[0])

        self.energy = torch.tensor(energy[None, :])
        self.angle = ad[:, :, 0] / (1 + (self.energy / ad[:, :, 1]) ** 2) + ad[:, :, 2]
        self.angle = 1 / self.angle[:, :, None, None]

        sax, say = np.meshgrid(
            np.linspace(data_config['rangex'][0], data_config['rangex'][1], self.shapes[1]),
            np.linspace(data_config['rangey'][0], data_config['rangey'][1], self.shapes[2]),
            indexing='ij'
        )
        self.sax = torch.tensor(sax[np.newaxis, :, :]).to(device=self.device)
        self.say = torch.tensor(say[np.newaxis, :, :]).to(device=self.device)

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return self

    def _generate_weights_and_indices(self):
        indices = torch.randint(0, self.i1_data.shape[0], (self.batch_size, self.max_samples_per_example))
        weights = torch.ones((self.batch_size, self.max_samples_per_example), device=self.device)
        mask_vector = torch.rand(self.batch_size, device=self.device).unsqueeze(-1)
        mask = torch.rand_like(weights) < mask_vector
        weights = weights * mask.float()

        random_indices = torch.randint(0, self.max_samples_per_example, (self.batch_size,), device=self.device)
        weights[torch.arange(self.batch_size, device=self.device), random_indices] = 1

        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device)

        return weights, indices

    def get_index(self, indices_for_sum):
        i1_batch = self.i1_data[indices_for_sum]
        i2_batch = self.i2_data[indices_for_sum]
        spectrum = self.spectrum[indices_for_sum]
        ax = self.ax[indices_for_sum]
        ay = self.ay[indices_for_sum]
        angle = self.angle[indices_for_sum]

        return i1_batch, i2_batch, spectrum, ax, ay, angle

    def to_cuda(self, i1_batch, i2_batch, spectrum, ax, ay, angle):
        i1_batch = i1_batch.to(self.device, non_blocking=True)
        i2_batch = i2_batch.to(self.device, non_blocking=True)
        spectrum = spectrum.to(self.device, non_blocking=True)
        ax = ax.to(self.device, non_blocking=True)
        ay = ay.to(self.device, non_blocking=True)
        angle = angle.to(self.device, non_blocking=True)

        return i1_batch, i2_batch, spectrum, ax, ay, angle

    def summ(self, i1_batch, i2_batch, weights):
        i1_batch = (i1_batch * weights).sum(dim=1)
        i2_batch = (i2_batch * weights).sum(dim=1)
        return i1_batch, i2_batch

    def create_3d(self, spectrum, ax, ay, angle, weights, batch_size=6):
        def f(v):
            v.pow_(2)
            v.mul_(-0.5)
            v.exp_()
            return v

        output_shape = (spectrum.size(0), 1, *self.shapes)
        result = torch.empty(output_shape, device=spectrum.device, dtype=torch.float32)

        n_batches = (spectrum.size(0) + batch_size - 1) // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, spectrum.size(0))

            batch_spectrum = spectrum[start:end]
            batch_ax = ax[start:end]
            batch_ay = ay[start:end]
            batch_angle = angle[start:end]

            distribution = f((batch_ax[:, :, None] - self.sax)[:, :, None] * batch_angle) * f((batch_ay[:, :, None] - self.say)[:, :, None] * batch_angle)
            distribution /= distribution.sum(dim=(-2, -1), keepdim=True)
            distribution *= batch_spectrum[:, :, :, None, None]
            distribution /= torch.sum(distribution, dim=(-1, -2, -3), keepdim=True)

            distribution *= weights[start:end]
            distribution = distribution.sum(dim=1, keepdim=True)
            distribution = transform_distribution.forward(distribution)

            distribution = torch.nan_to_num(distribution, nan=0.0)
            result[start:end] = distribution

        return result

    def _prepare_batches(self, indices_for_sum, weights):
        i1_batch, i2_batch, spectrum, ax, ay, angle = self.get_index(indices_for_sum)
        i1_batch, i2_batch, spectrum, ax, ay, angle = self.to_cuda(i1_batch, i2_batch, spectrum, ax, ay, angle)
        dist = self.create_3d(spectrum, ax, ay, angle, weights)

        i1_batch, i2_batch = self.summ(i1_batch, i2_batch, weights)

        dist = torch.nan_to_num(dist, nan=0.0)
        return non_zero_norm(i1_batch), non_zero_norm(i2_batch), non_zero_norm(dist)

    def __iter__(self):
        iter_per_epoch = len(self.i1_data)
        if self.max_iter_per_epoch is not None:
            iter_per_epoch = self.batch_size * self.max_iter_per_epoch

        for start in range(0, iter_per_epoch, self.batch_size):
            weights, indices_for_sum = self._generate_weights_and_indices()
            yield self._prepare_batches(indices_for_sum, weights)
