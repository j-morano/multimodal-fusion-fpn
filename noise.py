from typing import Union
import unittest
import random

import torch
from torch import Tensor
from torch.nn import Module
from matplotlib import pyplot as plt
import numpy as np

from utils import get_factory_adder



add_noise_class, noise_classes = get_factory_adder()


@add_noise_class('gaussian')
class GaussianNoise(Module):
    def __init__(self, level: float):
        # Get mean and std from noise percentage using a basic formula
        self.mean = 0
        self.level = level

    def normalize_to(
        self,
        data: Tensor,
        min: Union[float, Tensor],
        max: Union[float, Tensor],
    ) -> Tensor:
        """Normalize data to range [min, max]."""
        data = data - data.min()
        data = data / (data.max() + 1e-8)
        data = data * (max - min) + min
        return data

    def __call__(self, data: Tensor) -> Tensor:
        data_min = data.min()
        data_max = data.max()
        std = self.level * (data_max - data_min)
        data = data + torch.randn_like(data)*std + self.mean
        # print(data.min(), data.max())
        # Keep range
        data = self.normalize_to(data, data_min, data_max)
        # print(data.min(), data.max())
        return data


@add_noise_class('masking')
class MaskingNoise(Module):
    def __init__(self, level: float, patch_size: float=0.1):
        self.level = level
        self.num_masks = int(self.level * 200)
        # Patch size is a percentage of the image height
        self.patch_size = patch_size

    def __call__(self, data: Tensor) -> Tensor:
        max_h, max_d, max_w = data.shape[-3:]
        # 10% of data range
        pct_10 = ((data.max() - data.min()) * 0.1).item()
        mean = data.mean()
        patch_size_h = max(int(max_h * self.patch_size), 1)
        patch_size_d = max(int(max_d * (1 - self.patch_size/2)), 1)
        patch_size_w = max(int(max_w * self.patch_size), 1)
        for _ in range(self.num_masks):
            approx_mean = mean + random.uniform(-pct_10, pct_10)
            # Get random coordinates
            w = np.random.randint(0, max_w - patch_size_w) #type:ignore
            h = np.random.randint(0, max_h - patch_size_h) #type:ignore
            if max_d > 1:
                d = np.random.randint(0, max_d - patch_size_d) #type:ignore
                # Apply mask
                data[:, :, h:h + patch_size_h, d:d + patch_size_d, w:w + patch_size_w] = approx_mean
            else:
                data[:, :, h:h + patch_size_h, :, w:w + patch_size_w] = approx_mean
        return data


class TestNoise(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Download image
        self.data = self.get_image('https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png')

    def get_image(self, image_link):
        import requests
        from PIL import Image
        from io import BytesIO
        response = requests.get(image_link)
        img = Image.open(BytesIO(response.content))
        # img = img.resize((100, 100))
        img = np.array(img)/255
        img = img[np.newaxis, np.newaxis, :, :, :]
        # Permute image
        img = np.transpose(img, (0, 1, 2, 4, 3))
        img = torch.tensor(img, dtype=torch.float32)
        print(img.shape)
        return img

    def _show_images(self, before, after):
        before = torch.sum(before[0, 0], dim=1).cpu().numpy()
        after = torch.sum(after[0, 0], dim=1).cpu().numpy()
        plt.subplot(1, 2, 1)
        plt.imshow(before)
        plt.subplot(1, 2, 2)
        plt.imshow(after)
        plt.show()

    def test_gaussian(self):
        # Create noise class
        noise = GaussianNoise(0.5)
        # Apply noise
        noisy_data = noise(self.data)
        # Plot data before and after noise
        self._show_images(self.data, noisy_data)

    def test_masking(self):
        # Create noise class
        noise = MaskingNoise(0.5)
        # Apply noise
        noisy_data = noise(self.data)
        # Plot data before and after noise
        self._show_images(self.data, noisy_data)
