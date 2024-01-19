from typing import List

from config import config
from common import (
    loss,
    metrics,
    mytransforms,
    dataloader_hrf_spec_aligned_seg,
    dataloader_vrc_vessel,
)

from utils import get_factory_adder



add_class, data_config_factory = get_factory_adder()


class mmetric:
    def __init__(self, key='Dice'):
        self.key = key

    def get(self, m: dict):
        return m[self.key]


class MMetric:
    def __init__(self, mm):
        self.mm = mm

    def build(self):
        return {
            self.mm: self
        }

    def get(self, m: dict):
        return m[self.mm]


class DefaultConfig:
    s_size = 32 # no. of B-scans
    w_size = 128 # B-scan width
    h_size = None  # B-scan height
    batch_size = 8
    rate_mode = 'minimum'
    monitor = 'Dice'
    monitor_mode = 'max'
    metrics_train = {}
    metrics_val = {}
    meta_metric_val = {}
    extra_transforms = []
    identity_transforms = {
        'normalization': [],
        'augmentation': [],
    }
    mask_variant = config.mask_variant
    transformations = {
        'image': {
            'normalization': [
                mytransforms.ZScoreNormalization(transform_keys=['image'],axis=(2,3)),
            ],
            'augmentation': [
                ### Extra augmentation
                # mytransforms.InPaintEnface(transform_keys=['image'],nframes=12,box_shape=[12,12], min_box_shape=[4,4]),
                mytransforms.MulNoiseAugmentation(transform_keys=['image'], dim=(1,), mu=1.0, sigma=0.05),
                ###
                mytransforms.AddNoiseAugmentation(transform_keys=['image'],dim=(0,),mu=0.0,sigma=0.2),
                mytransforms.ContrastAugmentation(transform_keys=['image'],min=0.9, max=1.1),
                mytransforms.IntensityShift(transform_keys=['image'],min=-0.2, max=0.2),
            ]
        },
        'slo': {
            'normalization': [
                mytransforms.ZScoreNormalization(transform_keys=['slo'],axis=(1,3)),
            ],
            'augmentation': [
                mytransforms.AddNoiseAugmentation(transform_keys=['slo'],dim=(0,),mu=0.0,sigma=0.12),
                mytransforms.ContrastAugmentation(transform_keys=['slo'],min=0.95, max=1.05),
                mytransforms.IntensityShift(transform_keys=['slo'],min=-0.07, max=0.07),
            ]
        },
        'mask': identity_transforms,
    }
    paths = {
        'oct': None,
        'split': None,
        'visits': None,
    }

    def get_criterion(self):
        losses = {
            'Dice Loss': loss.Dice_loss_jointv2(output_key='prediction', target_key='mask'),
            'BCE loss': loss.BCE_Lossv2(output_key='prediction', target_key='mask'),
        }
        return loss.Mix(losses=losses)

    def get_val_transforms(self) -> mytransforms.Compose:
        keys = list(self.transformations.keys())
        data_transform_val = []
        for k in self.transformations:
            data_transform_val += self.transformations[k]['normalization']
        data_transform_val += [
            mytransforms.NewRandomRelFit(
                transform_keys=keys,
                fit=[None, 16, None, 16]  # [None, 32, None, 32]
            ),
            mytransforms.ToTensorDict(transform_keys=keys)
        ]
        return mytransforms.Compose(data_transform_val)

    def get_transforms(self):
        keys = list(self.transformations.keys())

        s_size = self.s_size
        w_size = self.w_size
        h_size = self.h_size

        crop_transforms: List[mytransforms.Transform] = [
            mytransforms.NewRandomRelCrop(
                reference_key='image',
                transform_keys=keys,
                size=[None, s_size, h_size, w_size],
            )
        ]

        if config.crop.startswith('relative'):
            crop_transforms += [
                mytransforms.NewRandomRelSize(
                    transform_keys=[q for q in keys if q != config.fusion_modality],
                    fixed_size=[None, s_size, None, w_size],
                ),
                # Resize fusion modality image to a fixed size in order
                #   to build the batches.
                mytransforms.NewRandomRelSize(
                    transform_keys=[config.fusion_modality],
                    fixed_size=[None, 320, None, 128]
                )
            ]
        else:
            crop_transforms.append(
                mytransforms.NewRandomRelSize(
                    transform_keys=keys,
                    fixed_size=[None, s_size, None, w_size],
                )
            )

        data_transforms = []
        data_transforms += crop_transforms
        data_transforms.append(
            mytransforms.RandomRotation180(keys=keys)
        )

        # NOTE: This could help 'oct'-only models, so for a fair
        #   comparison with fusion models we should not use it.
        #   deactivated by default.
        if config.crop == 'oct' and config.rotation_augmentation:
            data_transforms.append(
                mytransforms.RandomEnfaceRotation(keys=keys)
            )
        data_transforms.append(
            mytransforms.RandomMirror(transform_keys=keys, dimensions=[1,3])
        )
        for k in self.transformations:
            data_transforms += self.transformations[k]['normalization']
            data_transforms += self.transformations[k]['augmentation']
        data_transforms += self.extra_transforms
        data_transforms += [
            mytransforms.ToTensorDict(transform_keys=keys)
        ]

        compose_transforms_train = mytransforms.Compose(data_transforms)

        compose_transforms_val = self.get_val_transforms()

        return compose_transforms_train, compose_transforms_val



@add_class('hrf')
class HRFConfig(DefaultConfig):
    paths = {
        'oct': '../Multimodal_GA_seg_HRF',
        'split': '../Multimodal_GA_seg_HRF/split_1_full.json',
        'visits': '../Multimodal_GA_seg_HRF/hrf_data.json',
    }

    metrics_train = {
        'Dice': metrics.Dice(output_key='prediction', target_key='mask'),
        'BCE': metrics.BCE(output_key='prediction', target_key='mask', slice=0),
    }

    metrics_val = {
        'Dice': metrics.Dice(output_key='prediction', target_key='mask'),
        'BCE': metrics.BCE(output_key='prediction', target_key='mask', slice=0),
        'Hausdorff': metrics.Hausdorff(output_key='prediction', target_key='mask', slice=0),
        'Hausdorff95': metrics.Hausdorff95(output_key='prediction', target_key='mask', slice=0),
    }
    rate_mode = 'minimum'

    def train_data(self, training_file_list, data_transform):
        return dataloader_hrf_spec_aligned_seg.HRF_SPEC_Dataset(
            path=self.paths['oct'],
            patients=training_file_list,
            multiplier=config.multiplier,
            patches_from_single_image=1,
            transforms=data_transform,
            get_spacing=True,
            visits_fn=self.paths['visits'],
            mask_variant=self.mask_variant,
        )

    def val_data(self, validation_file_list, data_transform_val):
        return dataloader_hrf_spec_aligned_seg.HRF_SPEC_Dataset(
            path=self.paths['oct'],
            patients=validation_file_list,
            multiplier=1,
            patches_from_single_image=1,
            transforms=data_transform_val,
            get_spacing=True,
            visits_fn=self.paths['visits'],
            mask_variant=self.mask_variant,
        )

    meta_metric_val = {'Dice': mmetric()}


@add_class('hrf_fusion')
class HRFFusionConfig(HRFConfig):
    transformations = {
        'image': DefaultConfig.transformations['image'],
        'mask': DefaultConfig.identity_transforms,
        config.fusion_modality: DefaultConfig.identity_transforms,
    }


@add_class('hrf_fusion_comp_only')
class HRFFusionCompOnlyConfig(HRFFusionConfig):
    transformations = {
        'image': DefaultConfig.identity_transforms,
        'mask': DefaultConfig.identity_transforms,
        config.fusion_modality: {
            'normalization': [
                mytransforms.ZScoreNormalization(transform_keys=[config.fusion_modality],axis=(1,3)),
            ],
            'augmentation': [
                ### Extra augmentation
                # mytransforms.InPaintEnface(transform_keys=['image'],nframes=12,box_shape=[12,12], min_box_shape=[4,4]),
                mytransforms.MulNoiseAugmentation(transform_keys=[config.fusion_modality], dim=(1,), mu=1.0, sigma=0.05),
                ###
                mytransforms.AddNoiseAugmentation(transform_keys=[config.fusion_modality],dim=(0,),mu=0.0,sigma=0.2),
                mytransforms.ContrastAugmentation(transform_keys=[config.fusion_modality],min=0.9, max=1.1),
                mytransforms.IntensityShift(transform_keys=[config.fusion_modality],min=-0.2, max=0.2),
            ]
        },
    }
    def get_val_transforms(self) -> mytransforms.Compose:
        keys = list(self.transformations.keys())
        data_transform_val = []
        for k in self.transformations:
            data_transform_val += self.transformations[k]['normalization']
        data_transform_val += [
            mytransforms.Disable(keys=['image']),
            mytransforms.NewRandomRelFit(
                transform_keys=[config.fusion_modality, 'mask'],
                fit=[None, 16, None, 16]  # [None, 32, None, 32]
            ),
            mytransforms.ToTensorDict(transform_keys=keys)
        ]
        return mytransforms.Compose(data_transform_val)

    def get_transforms(self):
        keys = list(self.transformations.keys())

        crop_transforms = [
            mytransforms.Disable(keys=['image']),
            mytransforms.NewRandomRelSize(
                transform_keys=[config.fusion_modality, 'mask'],
                fixed_size=[None, 512, None, 512]
            )
        ]

        data_transforms = []
        data_transforms += crop_transforms
        data_transforms.append(
            mytransforms.RandomRotation180(keys=keys)
        )

        data_transforms.append(
            mytransforms.RandomEnfaceRotation(
                keys=keys,
                range=(-90, 90),
                probablity=0.9,
            )
        )
        data_transforms.append(
            mytransforms.RandomMirror(transform_keys=keys, dimensions=[1,3])
        )
        for k in self.transformations:
            data_transforms += self.transformations[k]['normalization']
            data_transforms += self.transformations[k]['augmentation']
        data_transforms += self.extra_transforms
        data_transforms += [
            mytransforms.ToTensorDict(transform_keys=keys)
        ]

        compose_transforms_train = mytransforms.Compose(data_transforms)

        compose_transforms_val = self.get_val_transforms()

        return compose_transforms_train, compose_transforms_val


@add_class('vrc')
class VRCVConfig(HRFConfig):
    paths = {
        'oct': '../Multimodal_vrc_vessel',
        'split': '../Multimodal_vrc_vessel/split_i2.json',
        'visits': '../Multimodal_vrc_vessel/vrc_data.json',
    }
    preprocessed_bscan = None
    oct_variant = 'flat'

    def train_data(self, training_file_list, data_transform):
        return dataloader_vrc_vessel.VRC_Dataset(
            path=self.paths['oct'],
            patients=training_file_list,
            multiplier=config.multiplier,
            patches_from_single_image=1,
            transforms=data_transform,
            get_spacing=True,
            mask_variant=self.mask_variant,
            visits_fn=self.paths['visits'],
            preprocessed_bscan=self.preprocessed_bscan,
            oct_variant=self.oct_variant,
        )

    def val_data(self, validation_file_list, data_transform_val):
        return dataloader_vrc_vessel.VRC_Dataset(
            path=self.paths['oct'],
            patients=validation_file_list,
            multiplier=1,
            patches_from_single_image=1,
            transforms=data_transform_val,
            get_spacing=True,
            mask_variant=self.mask_variant,
            visits_fn=self.paths['visits'],
            preprocessed_bscan=self.preprocessed_bscan,
            oct_variant=self.oct_variant,
        )


@add_class('vrc_crop')
class VRCCropConfig(VRCVConfig):
    oct_variant = 'crop'


@add_class('vrc_lr2')
class VRCVLR2Config(VRCVConfig):
    """Like VRCV but with low resolution images for training. An extra
    transformation is added to downsample the images.
    """
    preprocessed_bscan = 'lr2'


@add_class('vrc_lr2_comp_only')
class VRCVLR2CompOnlyConfig(VRCVLR2Config, HRFFusionCompOnlyConfig):
    preprocessed_bscan = 'lr2'
