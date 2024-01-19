from typing import Optional, Union
import os
import json

import numpy as np
from skimage import io

from config import config
from common.abstract_dataloader import AbstractDataset



class HRF_SPEC_Dataset(AbstractDataset):
    def __init__(
        self,
        path,
        patients: Optional[Union[dict, list]]=None,
        multiplier=1,
        patches_from_single_image=1,
        transforms=None,
        mask_variant='faf',
        get_spacing=False,
        visits_fn: Optional[str]=None,
    ):
        super().__init__()
        self.path = path
        self.multiplier = multiplier
        self.patches_from_single_image = patches_from_single_image
        self.transforms = transforms
        self.mask_variant = mask_variant
        self.get_spacing = get_spacing
        self.patients = patients
        self.visits_fn = visits_fn
        assert self.patients is not None
        assert self.visits_fn is not None

        with open(self.visits_fn, 'r') as fp:
            self.visits = json.load(fp)

        self.dataset = self._make_abstract_dataset()

        self.real_length = len(self.dataset)
        print('scans:', str(self.real_length))

        self.patches_from_current_image = self.patches_from_single_image

    def _load(self, index):
        self.record = self.dataset[index].copy()

        image = np.load(
            os.path.join(
                self.record['path'],
                'bscan_flat.'+self.record['FileSetId']+'.npy'
            )
        ) # type: np.ndarray
        # Dimensions: front, top, right
        self.record['image'] = image[None]

        if self.get_spacing:
            self.record['spacing'] = np.load(
                os.path.join(
                    self.record['path'],
                    'spacing.'+self.record['FileSetId']+'.npy'
                )
            )
        if config.crop in ['oct']:
            prefix = 'preprocessed_images/bscan_size.'
        else:
            prefix = ''
        if self.mask_variant == 'faf':
            mask = io.imread(
                os.path.join(
                    self.record['path'],
                    'preprocessed_images/bscan_size.mask_faf.'+self.record['FileSetId']+'.png'
                )
            ) # type: np.ndarray
            mask = mask/256
        elif self.mask_variant == 'oct':
            mask = io.imread(
                os.path.join(
                    self.record['path'],
                    'mask_oct.'+self.record['FileSetId']+'.png'
                )
            ) # type: np.ndarray
            mask = mask/256
        else:
            raise ValueError('Unknown mask variant')
        # Apply threshold
        mask = np.where(mask>=0.5, 1., 0.)
        self.record['mask'] = mask[None,:,None,:]

        if config.fusion_modality == 'slo':
            slo = io.imread(
                os.path.join(
                    self.record['path'],
                    prefix+'slo.'+self.record['FileSetId']+'.png'
                )
            ) # type: np.ndarray
            slo = slo/256
            self.record['slo'] = slo[None,:,None,:]
        elif config.fusion_modality == 'faf':
            faf = io.imread(
                os.path.join(
                    self.record['path'],
                    prefix+'faf.'+self.record['FileSetId']+'.png'
                )
            ) # type: np.ndarray
            faf = 1 - faf
            faf = faf/256
            self.record['faf'] = faf[None,:,None,:]
        else:
            raise ValueError('Unknown fusion modality')
