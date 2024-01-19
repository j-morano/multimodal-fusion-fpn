import os
import configparser

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from config import config
from models.fpn.unets3D import ModifiedUnet3D
from models.fpn.unets2D import ModifiedUnet2DLevel5
from models.fpn.fusion3D2D import ModifiedUnet3D2DLevel5

from utils import get_factory_adder


add_class, factory_classes = get_factory_adder()



class FPNConfig(nn.Module):
    def __init__(self):
        super().__init__()
        config_filename='modifiedUnet3D_red-convPlusFully_dropout00'
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join('models', 'fpn', config_filename + '.ini'))


@add_class
class FPN(FPNConfig):
    def __init__(self):
        super().__init__()
        self.resensnet = ModifiedUnet3D(self.config)

    def last_activation(self, x): return torch.sigmoid(x)

    def forward(self, x):
        # Z x W x H
        oct = x['image'].permute(0,1,2,4,3)
        oct_seg = self.resensnet(oct)
        oct_seg = oct_seg.permute(0,1,2,4,3)
        seg = self.last_activation(oct_seg)
        return {
            'prediction': seg,
        }


@add_class
class FPNRegression(FPN):
    def last_activation(self, x): return x


@add_class
class FPNClassification(FPN, FPNConfig):
    def __init__(self):
        FPNConfig.__init__(self)
        self.resensnet = ModifiedUnet3D(self.config, classification=True)
        self.one_one = nn.Conv3d(
            256,
            config.number_of_outputs,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def last_activation(self, _x):
        raise NotImplementedError

    def forward(self, x):
        # Z x W x H
        oct = x['image'].permute(0,1,2,4,3)
        pred = self.resensnet(oct)
        pred = self.one_one(pred)
        pred = self.adaptive_pool(pred).squeeze(-1).squeeze(-1).squeeze(-1)
        pred = torch.softmax(pred, dim=-1)
        return {
            'prediction': pred,
        }



@add_class
class FPNHybridFusion(FPNConfig):
    """Class for the FPN model for hybrid fusion. It is composed
    of a 3D encoder, a 2D encoder, and a common 2D decoder. The features
    of the 2 encoders are concatenated and fed to the decoder at the
    different levels. For the 3D encoder, we use 3D->2D blocks to
    project the 3D features to 2D feature space.
    The complementary modality can have different resolutions at the
    input. If the en-face resolution of the complementary modality is
    different from the primary modality, we interpolate the features
    when necessary. If the en-face resolution is the same, we do not
    need to interpolate.
    With the '2d' option, we interpolate 2d features to 3d en-face
    resolution. In this case, there is no need to interpolate the
    output.
    """
    def __init__(self):
        super().__init__()
        if 'relative_2d' in config.crop:
            self.interpolate = '2d'
        else:
            self.interpolate = None
        if 'max' in config.crop and self.interpolate is not None:
            self.interpolate += '_max'
        self.resensnet = ModifiedUnet3D2DLevel5(self.config, self.interpolate)

    def last_activation(self, x): return torch.sigmoid(x)

    def forward(self, x):
        # Z x W x H
        oct = x['image'].permute(0,1,2,4,3)
        slo = x[config.fusion_modality][:,:,:,0,:]
        oct_seg = self.resensnet(oct, slo)
        oct_seg = oct_seg.permute(0,1,2,4,3)
        seg = self.last_activation(oct_seg)
        return {
            'prediction': seg,
        }



@add_class
class FPNHybridFusionRegression(FPNHybridFusion):
    def last_activation(self, x): return x



@add_class
class FPN2D(FPNConfig):
    def __init__(self):
        super().__init__()
        self.resensnet = ModifiedUnet2DLevel5(self.config)

    def forward(self, x):
        # Z x W x H
        fused = x[config.fusion_modality][:,:,:,0,:]
        seg = self.resensnet(fused)
        seg = seg.permute(0,1,2,4,3)
        seg = torch.sigmoid(seg)
        if seg.shape != x['mask'].shape:
            seg = F.interpolate(
                seg, size=x['mask'].shape[2:], mode='trilinear',
            )
        return {
            'prediction': seg,
        }


@add_class
class FPNLateFusion(FPNConfig):
    """Class for the FPN model with late fusion. The difference
    with the FPNLate class is that the 2D part is also based on
    the FPN architecture. The overall idea is to process the
    images from the different modalities separately using 2 different
    nets, and then fuse the final features of each part to obtain the
    final segmentation.
    The input, as for FPNHybridFusion, the en-face resolution of
    the complementary modality can be the same as the primary modality
    or different. If it is different, we interpolate the features.
    -> See the FPNHybridFusion class for more details on the
    interpolation.
    """
    def __init__(self):
        super().__init__()
        self.resensnet3d = ModifiedUnet3D(self.config)
        self.resensnet2d = ModifiedUnet2DLevel5(self.config, output_features=True)
        # Ensure to do not use sigmoid in the subnets
        self.resensnet3d.use_1x1 = False
        self.fusion_module = nn.Conv3d(32, config.number_of_outputs, (1,1,1))
        if 'relative_2d' in config.crop:
            self.interpolate = '2d'
        else:
            self.interpolate = None
        if 'max' in config.crop and self.interpolate is not None:
            self.interpolate += '_max'

    def last_activation(self, x): return torch.sigmoid(x)

    def forward(self, x):
        oct = x['image'].permute(0,1,2,4,3)
        oct_seg = self.resensnet3d(oct)
        # Permute to have the dimensions in the form:
        #   (batch, channels, # of B-scans, B-scan height, B-scan width)
        #   or (another way to see it):
        #   (batch, channels, en-face height, depth, en-face width)
        oct_seg = oct_seg.permute(0,1,2,4,3)

        fused = x[config.fusion_modality][:,:,:,0,:]
        fused_seg = self.resensnet2d(fused)
        fused_seg = fused_seg.permute(0,1,2,4,3)
        if self.interpolate == '2d':
            fused_seg = F.interpolate(
                fused_seg, size=oct_seg.shape[2:], mode='trilinear',
            )
        elif self.interpolate == '2d_max':
            fused_seg = F.adaptive_max_pool3d(
                fused_seg, output_size=oct_seg.shape[2:]
            )

        seg = self.fuse_features(oct_seg, fused_seg)
        seg = self.last_activation(seg)
        # out_features = torch.cat([oct_seg.clone(), fused_seg.clone()], dim=1)
        # Deactivating the gradient for the features
        # out_features = out_features.detach()
        return {
            'prediction': seg,
            # 'out_features': out_features,
        }

    def fuse_features(self, oct_seg: Tensor, fused_seg: Tensor):
        oct_fused_seg = torch.cat([oct_seg, fused_seg], dim=1)
        seg = self.fusion_module(oct_fused_seg)
        return seg



@add_class
class FPNLateFusionRegression(FPNLateFusion):
    def last_activation(self, x): return x
