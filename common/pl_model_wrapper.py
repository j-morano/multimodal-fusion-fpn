from os.path import join
import gc
import time
import json
from typing import Dict, List, Tuple

import torch
import pytorch_lightning as pl
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.morphology import binary_erosion
from skimage.morphology import disk
from utils import MonitorLearning
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from utils import normalize_data



matplotlib.use('Agg')
pyplot_colors = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf'
]


def text_phantom(text: str, size: Tuple[int, int]) -> np.ndarray:
    w_size, h_size = size

    # Create font
    pil_font = ImageFont.truetype(
        "./assets/FiraCode-Medium.ttf",
        size=18,
        encoding="unic"
    )
    text_width, text_height = pil_font.getsize(text)  # type: ignore

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', size, (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((w_size - text_width) // 2,
              (h_size - text_height) // 2)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    canvas = (255 - np.asarray(canvas)) / 255.0  # type: ignore
    # Convert to grayscale
    canvas = np.mean(canvas, axis=2)
    return canvas


def compute_text_labels(
    keys: List[str],
    size: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    text_labels = {}
    for key in keys:
        text_labels[key] = text_phantom(key, size)
    return text_labels


class StaticFactory:
    classes = []

    def __init__(self, classes=None):
        if classes is not None:
            self.classes = classes
        self.classes_names = {
            class_.__name__: class_
            for class_
            in self.classes
        }

    def create_class(self, class_name, *args, **kwargs):
        return self.classes_names[class_name](*args, **kwargs)


class ReleaseMemCache:
    def __call__(self):
        # Force garbage collection and cache memory freeing
        gc.collect()
        torch.cuda.empty_cache()


class DoNotReleaseMemCache:
    def __call__(self):
        pass


class MemCacheStrategies(StaticFactory):
    classes = [ReleaseMemCache, DoNotReleaseMemCache]


class Model(pl.LightningModule):
    def __init__(
        self,
        model,
        losses,
        training_metrics,
        metrics,
        metametrics,
        optim,
        force_mem_cache_release="DoNotReleaseMemCache",
        validation=None,
        _log_file=None,
        model_path: str='',
    ):
        super().__init__()
        self.model = model
        self.loss = losses
        self.metrics = metrics
        self.metametrics = metametrics
        self.optim = optim
        self.training_metrics = training_metrics
        self.validation = validation
        print(self.validation)
        self.force_mem_cache_release = MemCacheStrategies().create_class(
            force_mem_cache_release
        )
        self.monitor_learning = MonitorLearning()
        self.curves = {}
        self.metric_colors = {}
        self.metric_figures = set()
        if self.training_metrics is not None:
            for tm in self.training_metrics.keys():
                self.metric_figures.add(tm)
                self.curves[f'{tm} (train)'] = []
                if tm not in self.metric_colors:
                    self.metric_colors[tm] = pyplot_colors.pop(0)
        if self.metrics is not None:
            for vm in self.metrics.keys():
                self.metric_figures.add(vm)
                self.curves[f'{vm} (val)'] = []
                if vm not in self.metric_colors:
                    self.metric_colors[vm] = pyplot_colors.pop(0)
        self.model_path = model_path
        ## For debugging
        self.image_keys = ['weight', 'mask', 'prediction', 'image', 'slo', 'faf']
        self.text_labels = compute_text_labels(self.image_keys, (256, 32))

    def forward(self, x, **kwargs):  # type: ignore
        self.force_mem_cache_release()
        prediction = self.model(x, **kwargs)
        if (
            self.validation is not None
            or (
                self.validation is None
                and self.monitor_learning.is_save_time()
            )
        ):
            x['prediction'] = prediction['prediction']
            self.debug_batch(x)
        return prediction

    def debug_batch(self, batch: dict):
        images = {}
        mask = None
        batch_size = batch['prediction'].shape[0]
        labels = None
        borders_key = 'weight' if 'weight' in batch.keys() else 'mask'
        for b_i in range(batch_size):
            bin_mask_borders = None
            for k in self.image_keys:
                if k not in batch:
                    continue
                if k == 'mask':
                    order = 0
                else:
                    order = 1
                image = batch[k]
                try:
                    image = image.detach().cpu().numpy()[b_i,0,:,:,:].sum(axis=1)
                except IndexError:
                    continue
                image = resize(
                    image,
                    (256,256),
                    order=order,
                    preserve_range=True,
                    anti_aliasing=False
                )  # type: np.ndarray
                image = normalize_data(image)
                if k == borders_key:
                    order = 0
                    mask = image
                    bin_mask = (mask > 0.5)
                    bin_mask_borders = (
                        bin_mask.astype(float)
                        - binary_erosion(bin_mask, disk(2)).astype(float)
                    )
                else:
                    if bin_mask_borders is not None:
                        image[bin_mask_borders == 1] = 1
                try:
                    images[b_i] = np.concatenate([images[b_i], image], axis=1)
                except KeyError:
                    images[b_i] = image
                if b_i == 0:  # We need only one row of labels
                    if labels is None:
                        labels = self.text_labels[k]
                    else:
                        labels = np.concatenate([labels, self.text_labels[k]], axis=1)
                print(k, batch[k].shape, torch.unique(batch[k]))
        all_images = np.concatenate([v for _, v in images.items()], axis=0)
        assert labels is not None
        # Write key 'k' in the image
        all_images = np.concatenate([labels, all_images], axis=0)  # type: np.ndarray
        current_ms = str(int(time.time()*1000))
        save_path = None
        if self.validation is not None:
            save_path = self.validation
            current_ms = batch['FileSetId'][0]
        else:
            save_path = join(self.model_path, 'images')
        # Concatenate the last 6 digits of the FileSetId of all the
        #  items in the batch
        name = ''
        for fsid in batch['FileSetId']:
            name = name+'_'+fsid[-6:]
        name = name[1:]
        current_ms = current_ms + '.' + name

        assert save_path is not None
        io.imsave(
            join(save_path, f'{current_ms}.png'),
            (all_images * 255).astype(np.uint8)
        )

    def training_step(self, batch, _batch_idx):  # type: ignore
        res = self(batch)
        loss, values = self.loss(batch, res)

        for k in values:
            self.log('Training/'+str(k), values[k].item(), on_step=True,on_epoch=False)

        with torch.no_grad():
            for k in self.training_metrics:
                self.training_metrics[k].update(batch,res)

        return loss

    def training_epoch_end(self, _outputs) -> None:  # type: ignore
        metric_results = {
            k: self.training_metrics[k].get()
            for k
            in self.training_metrics
        }

        if self.training_metrics is not None:
            for k in self.training_metrics:
                self.log('Training/' + str(k), metric_results[k], on_epoch=True)
                self.training_metrics[k].reset()
                self.curves[k+' (train)'].append(metric_results[k])

        # Save matplotlib plot with all the curves
        # Subplots with one row per metric
        fig, axs = plt.subplots(
            len(self.metric_figures),
            1,
            figsize=(20, 10*len(self.metric_figures)),
            squeeze=False,  # Always return a 2D array of subplots
        )
        for i, mf in enumerate(self.metric_figures):
            for k in self.curves:
                if mf not in k:
                    continue
                if '(val)' in k:
                    linestyle = '--'
                else:
                    linestyle = '-'
                axs[i,0].plot(
                    self.curves[k],
                    label=k,
                    linestyle=linestyle,
                    color=self.metric_colors[k.split(' ')[0]]
                )
            axs[i,0].legend()
            axs[i,0].set_title(mf)
            axs[i,0].grid(axis='y')
        fig.savefig(join(self.model_path, 'curves.svg'), bbox_inches='tight')  #type: ignore
        with open(join(self.model_path, 'curves.json'), 'w') as f:
            json.dump(self.curves, f)
        plt.close(fig)

        del metric_results
        gc.collect()
        torch.cuda.empty_cache()

    def validation_step(self, batch, _batch_idx):  # type: ignore
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            res = self(batch)

            for k in self.metrics:
                self.metrics[k].update(batch,res)

    def validation_epoch_end(self, _validation_step_outputs):  # type: ignore
        metric_results = {k:self.metrics[k].get() for k in self.metrics}

        for k in self.metrics:
            self.log('Validation/'+str(k), metric_results[k], on_epoch=True)
            self.metrics[k].reset()
            self.curves[k+' (val)'].append(metric_results[k])

        if self.metametrics is not None:
            for k in self.metametrics:
                self.log(str(k), self.metametrics[k].get(metric_results), on_epoch=True)

        del metric_results
        gc.collect()
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        return self.optim

