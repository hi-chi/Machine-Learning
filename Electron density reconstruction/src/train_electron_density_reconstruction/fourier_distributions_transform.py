from module.constant import device

import numpy as np
import torch
from scipy.signal import convolve2d

DISABLE_PROCESSING = False


class TransformDistribution:
    filter_kernel = np.array([
        [0, 0, -0.5, 0, 0],
        [0, -1, -1, -1, 0],
        [-0.5, -1, 12, -1, -0.5],
        [0, -1, -1, -1, 0],
        [0, 0, -0.5, 0, 0]
    ])

    filter_kernel = filter_kernel / filter_kernel.sum()

    def __init__(self):
        self.torch_filter_kernel = torch.tensor(self.filter_kernel, device=device)

    def forward(self, signal):
        if DISABLE_PROCESSING:
            return signal


        shift_y = self.torch_filter_kernel.shape[0] // 2
        shift_x = self.torch_filter_kernel.shape[1] // 2

        fft_shape = signal.shape[-2:]

        fft_signal = torch.fft.fft2(signal, dim=(-2, -1))
        fft_kernel = torch.fft.fft2(self.torch_filter_kernel, s=fft_shape)

        while fft_kernel.dim() < signal.dim():
            fft_kernel = fft_kernel.unsqueeze(0)
        fft_kernel = fft_kernel.expand_as(fft_signal)

        epsilon = 1e-8
        fft_recovered = fft_signal / (fft_kernel + epsilon)

        fft_recovered = torch.fft.ifft2(fft_recovered, dim=(-2, -1)).real

        fft_recovered = torch.roll(
            fft_recovered, shifts=(shift_y, shift_x), dims=(-2, -1)
        )

        return torch.sqrt(fft_recovered)

    def inverse(self, filtered_signal):
        if DISABLE_PROCESSING:
            return filtered_signal

        filtered_signal = filtered_signal ** 2

        result = np.zeros_like(filtered_signal)

        for i in range(filtered_signal.shape[0]):
            for j in range(filtered_signal.shape[2]):
                result[i, 0, j] = convolve2d(filtered_signal[i, 0, j], self.filter_kernel, mode='same')

        return result

transform_distribution = TransformDistribution()
