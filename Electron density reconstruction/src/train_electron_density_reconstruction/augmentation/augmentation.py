import random

import numpy as np
import torch
import torch.nn.functional as F

DISABLE_PROCESSING = False


class EmptyAugmentation:
    def __call__(self, images):
        return images


class CustomAugmentation:

    def __init__(self):
        self.noise_stats = {
            'all': 0,
            'gaussian_blur': 0,
            'perlin_1x1': 0,
            'perlin_small': 0,
            'perlin_big': 0,
            'gaussian_noise': 0,
            'uniform_noise': 0,
            'salt_pepper': 0,
            'lines': 0,
            'mega_noise': 0,
            'const': 0,
        }

    def apply_noise(self, image, noise, amax, coef):
        noise_max = noise.amax(dim=(1, 2, 3), keepdim=True) + 1e-8
        return image + (amax / noise_max) * coef * noise * torch.rand_like(image[:, :1, :1, :1])

    def gaussian_blur(self, input_tensor, kernel_size=3, sigma=1.0):
        kernel = torch.exp(-(torch.arange(kernel_size) - (kernel_size // 2)) ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, 1) * kernel.view(1, 1, 1, kernel_size)
        kernel = kernel.repeat(input_tensor.size(1), 1, 1, 1)
        kernel = kernel.to(input_tensor.device)

        return F.conv2d(input_tensor, kernel, padding=kernel_size // 2, groups=input_tensor.size(1))

    def generate_line(self, image):
        noise = torch.ones_like(image)

        v = torch.ones((image.shape[2]))
        v[torch.rand((image.shape[2])) < 0.95] = 0
        noise = noise - v[None, None, :, None].to(image.device)

        v = torch.ones((image.shape[3]))
        v[torch.rand((image.shape[3])) < 0.95] = 0
        noise = noise - v[None, None, None, :].to(image.device)
        noise = torch.clamp(noise, random.random())
        kernel_size = 2 * torch.randint(0, 4, (1,)).item() + 3

        noise = self.gaussian_blur(noise, kernel_size, sigma=1)
        return image * noise

    def perlin_noise_2d_torch_batch(self, batch_size, shape, periods):
        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)

        def lerp(a, b, x):
            return a + x * (b - a)

        gx, gy = shape
        px, py = periods

        grid_x, grid_y = gx // px, gy // py
        grad_x = torch.rand(batch_size, grid_x + 1, grid_y + 1, device=torch.device('cuda')) * 2 - 1
        grad_y = torch.rand(batch_size, grid_x + 1, grid_y + 1, device=torch.device('cuda')) * 2 - 1

        xx, yy = torch.meshgrid(
            torch.linspace(0, px, gx, dtype=torch.float32, device=grad_x.device),
            torch.linspace(0, py, gy, dtype=torch.float32, device=grad_y.device),
            indexing="ij"
        )
        xx, yy = xx.unsqueeze(0).expand(batch_size, -1, -1), yy.unsqueeze(0).expand(batch_size, -1, -1)

        x0, y0 = xx.floor().long(), yy.floor().long()
        x1, y1 = (x0 + 1).clamp(max=grid_x), (y0 + 1).clamp(max=grid_y)

        xs, ys = fade(xx - x0.float()), fade(yy - y0.float())

        batch_indices = torch.arange(batch_size, device=grad_x.device).view(batch_size, 1, 1)

        n00 = grad_x[batch_indices, x0, y0] * (xx - x0.float()) + grad_y[batch_indices, x0, y0] * (yy - y0.float())
        n10 = grad_x[batch_indices, x1, y0] * (xx - x1.float()) + grad_y[batch_indices, x1, y0] * (yy - y0.float())
        n01 = grad_x[batch_indices, x0, y1] * (xx - x0.float()) + grad_y[batch_indices, x0, y1] * (yy - y1.float())
        n11 = grad_x[batch_indices, x1, y1] * (xx - x1.float()) + grad_y[batch_indices, x1, y1] * (yy - y1.float())

        nx0 = lerp(n00, n10, xs)
        nx1 = lerp(n01, n11, xs)
        return lerp(nx0, nx1, ys)

    def gen_gaussian_noise(self, images, noise_shape):
        return torch.randn(noise_shape, device=images.device)

    def gen_uniform_noise(self, images, noise_shape):
        return torch.rand(noise_shape, device=images.device) * 2 - 1

    def apply_salt_pepper_noise(self, images, noise_level, noise_shape):
        salt_pepper = torch.rand(noise_shape, device=images.device)
        images[salt_pepper < noise_level * 0.5] = 1.0
        images[salt_pepper > 1 - noise_level * 0.5] = 0.0
        return images

    def dice_roll(self, chance=1., flag=False):
        return flag or random.random() < chance

    def generate_exponential_distribution(self, shape: tuple, lam: float = 20.0, device: str = 'cpu') -> torch.Tensor:
        return torch.clip(-lam * torch.log(torch.rand(shape, device=device)), 0, 1)

    def generate_exponential_number(self, lam: float = 20.0) -> float:
        return np.clip(-lam * np.log(random.random()), 0, 1)

    def reset_stats(self):
        self.noise_stats = {
            'all': 0,
            'gaussian_blur': 0,
            'perlin_1x1': 0,
            'perlin_small': 0,
            'perlin_big': 0,
            'gaussian_noise': 0,
            'uniform_noise': 0,
            'salt_pepper': 0,
            'lines': 0,
            'mega_noise': 0,
            'const': 0,
        }

    def mega_noise(self, noise_shape, coef):
        n1 = torch.ones((noise_shape[0], 1, 4, 12))
        n2 = torch.ones((noise_shape[0], 1, 8, 24))
        n1[torch.rand((noise_shape[0], 1, 4, 12)) > coef] = 0
        n2[torch.rand((noise_shape[0], 1, 8, 24)) > coef] = 0
        n1 = F.interpolate(n1, size=noise_shape[2:], mode='bilinear', align_corners=False)
        n2 = F.interpolate(n2, size=noise_shape[2:], mode='bilinear', align_corners=False)
        return torch.clip(n1 + n2, 0, 1)

    def augment_data(self, images, **kwargs):
        if DISABLE_PROCESSING: return images


        if not hasattr(self, 'noise_stats'):
            self.noise_stats = {
                'all': 0,
                'gaussian_blur': 0,
                'perlin_1x1': 0,
                'perlin_small': 0,
                'perlin_big': 0,
                'gaussian_noise': 0,
                'uniform_noise': 0,
                'salt_pepper': 0,
                'lines': 0,
                'mega_noise': 0,
                'const': 0,
            }
        self.noise_stats['all'] += 1

        noise_shape = images.shape

        epoch = kwargs.get('epoch', None)
        if epoch is not None:
            epoch_coef = epoch / kwargs['epoch_norm']
        else:
            epoch_coef = 0.001

        noise_amplidute =  self.generate_exponential_number(lam=epoch_coef)
        noise_chance = epoch_coef
        amax = torch.amax(images, dim=(1, 2, 3))[:, None, None, None].to(images.device)

        run_all = False
        common_chance = 1 * noise_chance
        perlin_chance = 1 * noise_chance

        vcommon = 0.10 * noise_amplidute

        vperlin_1x1 = 0.3 * noise_amplidute
        vperlin_small = 0.3 * noise_amplidute
        vperlin_big = 0.3 * noise_amplidute

        if self.dice_roll(common_chance, run_all):
            images = self.gaussian_blur(images, kernel_size=2 * random.randint(1, 2) + 1, sigma=random.randint(1, 3))
            self.noise_stats['gaussian_blur'] += 1

        if self.dice_roll(perlin_chance, run_all):
            if random.random() < 0.5:
                noise = self.perlin_noise_2d_torch_batch(
                    images.shape[0], images.shape[2:], periods=(1, 1)
                )
                noise[noise < 0] = 0
                noise = noise[:, None] * torch.amax(noise, dim=(1, 2))[:, None, None, None]
                noise = noise.to(device=images.device)
                images = self.apply_noise(images, noise, amax, vperlin_1x1)
                self.noise_stats['perlin_1x1'] += 1
            else:
                noise = self.perlin_noise_2d_torch_batch(
                    images.shape[0], images.shape[2:], periods=(random.randint(1, 3), random.randint(1, 3))
                )
                noise[noise < 0] = 0
                noise = noise[:, None] * torch.amax(noise, dim=(1, 2))[:, None, None, None]
                noise = noise.to(device=images.device)
                images = self.apply_noise(images, noise, amax, vperlin_small)
                self.noise_stats['perlin_small'] += 1

        if self.dice_roll(perlin_chance, run_all):
            r = (random.randint(1, 6), random.randint(1, 10))
            noise = self.perlin_noise_2d_torch_batch(
                images.shape[0], images.shape[2:], periods=r
            )
            noise[noise < 0] = 0
            noise = noise[:, None] * torch.amax(noise, dim=(1, 2))[:, None, None, None]
            noise = noise.to(device=images.device)
            images = self.apply_noise(images, noise, amax, vperlin_big)
            self.noise_stats['perlin_big'] += 1

        if self.dice_roll(common_chance, run_all):
            noise = self.gen_gaussian_noise(images, noise_shape).to(device=images.device)
            images = self.apply_noise(images, noise, amax, vcommon)
            self.noise_stats['gaussian_noise'] += 1

        if self.dice_roll(common_chance, run_all):
            noise = self.gen_uniform_noise(images, noise_shape).to(device=images.device)
            images = self.apply_noise(images, noise, amax, vcommon)
            self.noise_stats['uniform_noise'] += 1

        if self.dice_roll(common_chance, run_all):
            images = self.apply_salt_pepper_noise(images, random.random() * vcommon, noise_shape)
            self.noise_stats['salt_pepper'] += 1

        if self.dice_roll(common_chance, run_all):
            images = self.generate_line(images)
            self.noise_stats['lines'] += 1

        if self.dice_roll(common_chance, run_all):
            noise = torch.ones_like(images).to(device=images.device)
            images = self.apply_noise(images, noise, amax, 10 * vcommon)
            self.noise_stats['const'] += 1

        if self.dice_roll(common_chance, run_all):
             images = self.gaussian_blur(images, kernel_size=2 * random.randint(1, 2) + 1, sigma=random.randint(1, 3))
             self.noise_stats['gaussian_blur'] += 1

        return images

    def __call__(self, images, **kwargs):
        return self.augment_data(images, **kwargs)

