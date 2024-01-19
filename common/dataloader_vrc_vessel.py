from pathlib import Path
from typing import Optional, Union
import os
from os.path import join
import json

import numpy as np

from config import config
from common.abstract_dataloader import AbstractDataset
from skimage import io



class VRC_Dataset(AbstractDataset):
    def __init__(
        self,
        path,
        patients=None,
        multiplier=1,
        patches_from_single_image=1,
        transforms=None,
        mask_variant=None,
        get_spacing=False,
        visits_fn: Optional[str]=None,
        preprocessed_bscan: Optional[str]=None,
        oct_variant: str='flat',
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
        self.preprocessed_bscan = preprocessed_bscan
        self.oct_variant = oct_variant

        assert self.patients is not None
        assert self.visits_fn is not None

        with open(self.visits_fn, 'r') as fp:
            self.visits = json.load(fp)

        self.dataset = self._make_abstract_dataset()

        self.real_length = len(self.dataset)
        print('scans:', str(self.real_length))

        self.patches_from_current_image = self.patches_from_single_image

    def _make_dataset_ids(self, ids: list) -> list:
        raise NotImplementedError

    def _make_dataset(self, patients: Union[dict, list]) -> list:
        dataset = []

        for k in patients:
            for visit in self.visits[k]:
                record = {}
                record['path'] = join(self.path, k)
                record['FileSetId'] = visit['FileSetId']
                record['VRCPatId'] = k

                dataset.append(record)

        return dataset

    def _load(self, index):
        self.record = self.dataset[index].copy()
        file_set_id = self.record['FileSetId']

        if self.oct_variant == 'flat':
            bscan_fn = 'bscan_flat.'+self.record['FileSetId']+'.npy'
            if self.preprocessed_bscan is not None:
                bscan_fn = (
                    'preprocessed_images/bscan_flat.'
                    + self.preprocessed_bscan
                    + '.'
                    + file_set_id
                    + '.npy'
                )

            image = np.load(
                os.path.join(
                    self.record['path'],
                    bscan_fn
                )
            )  # type: np.ndarray
            if self.get_spacing:
                self.record['spacing'] = np.load(
                    os.path.join(
                        self.record['path'],
                        'spacing.'+self.record['FileSetId']+'.npy'
                    )
                )

            if self.mask_variant == 'sq_proj_dil':
                mask_fn = 'bscan_size.vs_proj.dil.'+self.record['FileSetId']+'.png'
            else:
                mask_fn = 'vs.vmirror.'+self.record['FileSetId']+'.png'

            mask = io.imread(
                os.path.join(
                    self.record['path'],
                    'preprocessed_images',
                    mask_fn
                )
            )  # type: np.ndarray

            mask = mask/256
            # Apply threshold
            mask = mask > 0.5

            if config.crop in ['oct']:
                prefix = 'preprocessed_images/bscan_size.'
            else:
                prefix = ''
            slo = io.imread(
                os.path.join(
                    self.record['path'],
                    prefix+'slo.'+self.record['FileSetId']+'.png'
                )
            )  # type: np.ndarray
            slo = slo/256

            # Dimensions: front, top, right
            self.record['image'] = image[None]
            self.record['mask'] = mask[None,:,None,:]
            self.record['slo'] = slo[None,:,None,:]
        elif self.oct_variant == 'crop':
            path = Path(self.record['path'], 'cropped')
            image = np.load(
                join(
                    path,
                    f'bscan_crop.{file_set_id}.npy'
                )
            ) # type: np.ndarray
            if self.get_spacing:
                self.record['spacing'] = np.load(
                    join(
                        path.parent,
                        f'spacing.{file_set_id}.npy'
                    )
                )
            # dep('in __load: unique image values:', np.unique(image))
            # Dimensions: front, top, right
            self.record['image'] = image[None]

            mask = io.imread(
                join(
                    path,
                    f'vs_crop.{file_set_id}.png'
                )
            ) # type: np.ndarray
            mask = mask/256
            # Apply threshold
            mask = np.where(mask>=0.5, 1., 0.)
            self.record['mask'] = mask[None,:,None,:]

            slo = io.imread(
                join(
                    path,
                    f'slo_crop.{file_set_id}.png'
                )
            ) # type: np.ndarray
            slo = slo/256
            self.record['slo'] = slo[None,:,None,:]
        else:
            raise ValueError('Unknown OCT variant: '+self.oct_variant)
