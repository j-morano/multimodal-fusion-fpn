from typing import List, Tuple, Union

import numpy as np
import torch
import random
from scipy import ndimage
from skimage.transform import resize



class Transform:
    def __init__(self, transform_keys:list):
        self.transform_keys = transform_keys

    def __call__(self, _data: dict):
        raise NotImplementedError()


class Compose:
    def __init__(self, transforms:list):
        self.transforms = transforms

    def __call__(self, data:dict):
        results = data
        for t in self.transforms:
            results = t(data)
        return results


class NewRandomRelCrop(Transform):
    def __init__(self, reference_key: str, transform_keys: list, size: List[Union[int, None]]):
        super().__init__(transform_keys)
        self.size = size
        self.reference_key = reference_key

    def __call__(self, data: dict):
        rels = {}
        reference_shape = data[self.reference_key].shape
        for i, size in enumerate(self.size):
            if size is not None:
                if size > reference_shape[i]:
                    rand_start = 0
                else:
                    rand_start = random.randint(0, reference_shape[i]-size)
                rels[i] = {
                    'start': rand_start/reference_shape[i],
                    'size': size/reference_shape[i],
                }
        for k in self.transform_keys:
            starts_ends = []
            for i, size in enumerate(self.size):
                if data[k].shape[i] > 1 and size is not None:
                    abs_start = int(round(data[k].shape[i] * rels[i]['start']))
                    abs_size = int(round(data[k].shape[i] * rels[i]['size']))
                    abs_end = abs_start + abs_size
                else:
                    abs_start = 0
                    abs_end = data[k].shape[i]
                starts_ends.append((abs_start, abs_end))
            data[k] = data[k][
                starts_ends[0][0]:starts_ends[0][1],
                starts_ends[1][0]:starts_ends[1][1],
                starts_ends[2][0]:starts_ends[2][1],
                starts_ends[3][0]:starts_ends[3][1],
            ]
        return data


class NewRandomRelFit(Transform):
    def __init__(self, transform_keys: list, fit: List[Union[int, None]]):
        super().__init__(transform_keys)
        self.fit = fit

    def __call__(self, data: dict):
        for k in self.transform_keys:
            shapes = []
            i = 0
            for fit in self.fit:
                if fit is None:
                    shapes.append(data[k].shape[i])
                else:
                    # Get get the multiple of fit closest to
                    #   data[k].shape[i]
                    # Previous: fit_shape = (data[k].shape[i] // fit) * fit
                    fit_shape = int(round(data[k].shape[i] / fit) * fit)
                    # Do max to avoid 0 shape
                    shapes.append(max(fit, fit_shape))
                i += 1
            final_shape = (
                shapes[0],
                shapes[1],
                shapes[2],
                shapes[3],
            )
            # Avoid unnecessary resizing
            if final_shape == data[k].shape:
                continue
            if 'mask' in k:
                order = 0
            else:
                order = 1
            data[k] = resize(
                data[k],
                final_shape,
                order=order,
                preserve_range=True
            )
        return data


class NewRandomRelSize(Transform):
    def __init__(
        self,
        transform_keys: List[str],
        fixed_size: List[Union[int, None]]
    ):
        super().__init__(transform_keys)
        self.fixed_size = fixed_size

    def __call__(self, data: dict):
        for k in self.transform_keys:
            shapes = []
            i = 0
            for fixed_size in self.fixed_size:
                if fixed_size is None or data[k].shape[i] == 1:
                    shapes.append(data[k].shape[i])
                else:
                    shapes.append(fixed_size)
                i += 1
            final_shape = (
                shapes[0],
                shapes[1],
                shapes[2],
                shapes[3],
            )
            if final_shape == data[k].shape:
                continue
            if 'mask' in k:
                order = 0
            else:
                order = 1
            data[k] = resize(
                data[k],
                final_shape,
                order=order,
                preserve_range=True
            )
        return data


class RandomRotation180(Transform):
    def __init__(self, keys: list):
        self.keys = keys

    def __call__(self, data: dict):
        if random.random() > 0.5:
            for k in self.keys:
                data[k] = np.rot90(data[k], k=2, axes=(1, 3))
        return data


class Disable(Transform):
    def __init__(self, keys: list):
        self.keys = keys

    def __call__(self, data: dict):
        for k in self.keys:
            data[k] = np.zeros((1,) * len(data[k].shape))
        return data


class RandomEnfaceRotation(Transform):
    def __init__(
        self,
        keys: list,
        range: Tuple[float,float]=(-5.0,5.0),
        probablity: float=0.5
    ):
        self.keys = keys
        self.range = range
        self.probablity = probablity

    def __call__(self, data: dict):
        if random.random() < self.probablity:
            rotation = random.uniform(*self.range)
            for k in self.keys:
                if 'mask' in k:
                    order = 0
                else:
                    order = 1
                cval = 0.0
                data[k] = ndimage.rotate(
                    data[k],
                    rotation,
                    axes=(1, 3),
                    reshape=False,
                    order=order,
                    mode='constant',
                    cval=cval,
                    prefilter=False
                )
        return data


class RandomDownsample(Transform):
    def __init__(
        self,
        transform_keys: list,
        factor: int=2,
        dimensions: list=[1],
        start_slice: Union[int, None]=None
    ):
        super().__init__(transform_keys)
        self.factor = factor
        self.dimensions = dimensions
        self.start_slice = start_slice
        assert all([0 < dim < 4 for dim in dimensions])

    def __call__(self, data: dict):
        if self.start_slice is None:
            start_slice = random.randint(0, self.factor-1)
        else:
            start_slice = self.start_slice
        for k in self.transform_keys:
            for dim in self.dimensions:
                try:
                    if dim == 1:
                        data[k] = data[k][:, start_slice::self.factor, :, :]
                    elif dim == 2:
                        data[k] = data[k][:, :, start_slice::self.factor, :]
                    elif dim == 3:
                        data[k] = data[k][:, :, :, start_slice::self.factor]
                except IndexError:
                    pass
        return data


class RandomMirror(Transform):
    def __init__(self, transform_keys:list, dimensions:list):
        super(RandomMirror, self).__init__(transform_keys)
        self.dimensions = dimensions

    def flip(self, image, p):

        index = [slice(0, size) for size in image.shape]

        for i in self.dimensions:
            if p[i] < 0.5:
                index[i] = slice(-1, -image.shape[i] - 1, -1)

        index = tuple(index)

        return image[index].copy()

    def __call__(self, data:dict):
        if isinstance(data[self.transform_keys[0]], dict):
            dim = len(data[self.transform_keys[0]][0].shape)
        else:
            dim = len(data[self.transform_keys[0]].shape)

        p = np.random.random(dim)  # type:ignore

        for key in self.transform_keys:

            if not key in data:
                continue
            if isinstance(data[key],dict):
                for subkey in data[key]:
                    if data[key][subkey] is not None:
                        data[key][subkey] = self.flip(data[key][subkey],p)
            else:
                data[key] = self.flip(data[key],p)

        return data


class ZScoreNormalization(Transform):
    def __init__(self, transform_keys:list, axis):
        super(ZScoreNormalization, self).__init__(transform_keys)
        self.axis = axis

    def __call__(self, data:dict):
        #mean = 9310.
        #std = 8759
        for key in self.transform_keys:
            if isinstance(data[key],dict):
                for subkey in data[key]:
                    mean = data[key][subkey].mean(axis=self.axis, keepdims=True)
                    std = data[key][subkey].std(axis=self.axis, keepdims=True)
                    data[key][subkey] = (data[key][subkey] - mean) / (std)
            else:
                mean = data[key].mean(axis=self.axis, keepdims=True)
                std = data[key].std(axis=self.axis, keepdims=True)
                data[key] = (data[key] - mean) / (std+1e-8)

        return data


class IntensityShift(Transform):
    def __init__(self, transform_keys:list, min:float = -0.6, max:float = 0.6):
        super(IntensityShift, self).__init__(transform_keys)
        self.min = min
        self.max = max

    def __call__(self, data:dict):
        for key in self.transform_keys:
            if isinstance(data[key], dict):
                for subkey in data[key]:
                    data[key][subkey] = data[key][subkey] + random.uniform(self.min, self.max)
            else:
                data[key] = data[key] + random.uniform(self.min, self.max)

        return data


class ContrastAugmentation(Transform):
    def __init__(self, transform_keys: list, min: float = 0.6, max: float = 1.4):
        super(ContrastAugmentation, self).__init__(transform_keys)
        self.min = min
        self.max = max

    def __call__(self, data: dict):
        for key in self.transform_keys:
            if isinstance(data[key], dict):
                for subkey in data[key]:
                    data[key][subkey] = data[key][subkey] * random.uniform(self.min, self.max)

            else:
                data[key] = data[key] * random.uniform(self.min, self.max)

        return data

class AddNoiseAugmentation(Transform):
    def __init__(self, transform_keys: list, dim, mu: float = 0.0, sigma: float = 1.0):
        super(AddNoiseAugmentation, self).__init__(transform_keys)
        self.mu = mu
        self.sigma = sigma
        self.dim = dim

    def __call__(self, data: dict):
        for key in self.transform_keys:
            if isinstance(data[key],dict):
                for subkey in data[key]:
                    shape = [i if idx in self.dim else 1 for idx, i in enumerate(data[key][subkey].shape)]
                    noise = np.random.normal(self.mu, self.sigma, size=shape)  # type:ignore
                    data[key][subkey] = data[key][subkey] + noise  # random.uniform(self.min, self.max)
            else:
                shape = [i if idx in self.dim else 1 for idx, i in enumerate(data[key].shape)]
                noise = np.random.normal(self.mu, self.sigma, size=shape)  # type:ignore
                data[key] = data[key] + noise  # random.uniform(self.min, self.max)


class MulNoiseAugmentation(Transform):
    def __init__(
        self,
        transform_keys: list,
        dim,
        mu: float = 1.0,
        sigma: float = 1.0
    ):
        super(MulNoiseAugmentation, self).__init__(transform_keys)
        self.mu = mu
        self.sigma = sigma
        self.dim = dim

    def __call__(self, data: dict):
        for key in self.transform_keys:
            if isinstance(data[key],dict):
                for subkey in data[key]:
                    shape = [i if idx in self.dim else 1 for idx, i in enumerate(data[key][subkey].shape)]
                    noise = np.random.normal(self.mu, self.sigma, size=shape)  # type:ignore
                    data[key][subkey] = data[key][subkey] * noise  # random.uniform(self.min, self.max)
            else:
                shape = [i if idx in self.dim else 1 for idx, i in enumerate(data[key].shape)]
                noise = np.random.normal(self.mu, self.sigma, size=shape)  # type:ignore
                data[key] = data[key] * noise  # random.uniform(self.min, self.max)


class ToTensorDict(Transform):
    def __init__(self, transform_keys: list):
        super(ToTensorDict, self).__init__(transform_keys)

    def __call__(self, data: dict):
        for key in self.transform_keys:
            if not key in data:
                continue
            if isinstance(data[key],dict):
                for subkey in data[key]:
                    if data[key][subkey] is not None:
                        data[key][subkey] = torch.from_numpy(data[key][subkey]).float()
            else:
                data[key] = torch.from_numpy(data[key]).float()

        return data
