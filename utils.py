import os
from os.path import join
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import time

import numpy as np
import torch
from torch.nn import Module, Conv2d, ConvTranspose2d



class MonitorLearning:

    def __init__(self):
        self.minute = -1

    def is_save_time(self):
        # Save one image for debugging every minute
        now_minute = int(time.time() / 60)
        is_save_time = now_minute > self.minute
        if is_save_time:
            self.minute = now_minute
        return is_save_time


def array_to_cuda(array, device=None):
    if isinstance(array, torch.Tensor):
        if device is not None:
            array = array.to(device)
        else:
            array = array.cuda()
    elif isinstance(array, dict):
        for key in array:
            array[key] = array_to_cuda(array[key], device)
    elif isinstance(array, list):
        array = [array_to_cuda(a, device) for a in array]

    return array


def get_factory_adder() -> Tuple[Any, Dict[str, Any]]:
    """Get a function that adds a class to a list and the corresponding
    list. Useful for creating a factory with a list of classes. The
    intended use is as a decorator.
    You can also can specify a different name for the class in the list,
    to use it at creation time instead of the class name.
    Example:
        >>> add_class, classes_dict = get_factory_adder()
        >>> @add_class
        ... class A:
        ...     pass
        >>> @add_class('Cc')
        ... class C:
        ...     pass
    """
    classes_dict = {}
    def _add_class(class_: Any, name: Optional[str]=None) -> Any:
        if name is None:
            name = class_.__name__
        classes_dict[name] = class_
        return class_

    def add_class(class_: Any, name: Optional[str]=None) -> Any:
        if not callable(class_):
            name = class_
            def wrapper(class_: Any) -> Any:
                return _add_class(class_, name)
            return wrapper
        else:
            return _add_class(class_)

    return add_class, classes_dict


def count_parameters(module: Module):
    """Counts the number of learnable parameters in a Module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_conv2d(module: Module):
    """Counts the number of convolutions and transposed convolutions in
    a module.
    """
    return len([m for m in module.modules() if isinstance(m, Conv2d) or isinstance(m, ConvTranspose2d)])


def print_net_info(net: Module):
    """Prints the number of layers and the number of parameters of a
    network.
    """
    print('=====  Net info  =====')
    print('Layers:', count_conv2d(net))
    print('Parameters:', count_parameters(net))
    print('======================')


def normalize_data(data: np.array, zero_nans: bool=True) -> np.array:
    """ Normalize data to [0, 1] range."""
    if zero_nans:
        # Replace nans with zeros
        data = np.nan_to_num(data)
    # Normalize to [0, 1] range
    return (data - np.min(data)) / (np.max(data)+1e-10 - np.min(data))


def get_model_path(config, split_path, idx=None, return_split_name=False):
    model_path = os.path.join(
        config.models_path,
        config.training_dataset,
    )

    if config.training_dataset == 'vrc' and config.mask_variant != 'vs_proj':
        model_path = model_path + '_' + config.mask_variant

    split_name = Path(split_path).stem
    model_path = join(
        model_path,
        split_name,
    )
    if config.multiplier != 20:
        ratio_mul = "{}_mul-{}".format(config.data_ratio, config.multiplier)
    else:
        ratio_mul = "{}".format(config.data_ratio)
    model_path = join(model_path, ratio_mul)
    if idx is not None:
        model_path = os.path.join(model_path, str(idx))
    model_name = config.model
    if config.epochs != 40:
        model_name += '_'+str(config.epochs)
    if not config.legacy_path:
        if config.learning_rate != 0.01:
            model_name += '_'+str(config.learning_rate)
    if config.crop is not None:
        model_name += '_'+str(config.crop)
    if (
        config.fusion_modality is not None
        and config.use_complementary
    ):
        model_name += '-'+config.fusion_modality
    if config.model_weights is not None:
        weights = Path(config.model_weights).stem
        if weights == 'last':
            weights = Path(config.model_weights).parent.stem+'.ckpt'
        model_name += '__'+weights
    if config.suffix is not None and config.suffix != "":
        if config.legacy_path:
            model_name += config.suffix
        else:
            model_name += "-"+config.suffix
    model_path = join(model_path, model_name)
    if return_split_name:
        return model_path, split_name
    return model_path
