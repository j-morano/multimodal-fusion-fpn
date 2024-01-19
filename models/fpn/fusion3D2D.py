from typing import Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.fpn.components import Upsample_Custom3d_nearest, SegmentationNetwork
from config import config as global_config



class ModifiedUnet3D2D(SegmentationNetwork):
    '''
    ModifiedUnet3D U-Net architecture. Fully convolutional neural
    network with encoder/decoder architecture and skipping connections,
    with dropout in the intermediate layer.
    '''

    def __init__(
        self,
        config,
        interpolate: Union[str, None]=None,
        feature_fusion: str='concat',
    ):
        """
        Args:
            config: configuration object,
            interpolate: interpolation method for upsampling.
            feature_fusion: feature fusion method for skip connections.
                - choices: 'concat', 'add'
        """
        super().__init__(
            n_classes=global_config.number_of_outputs,
            is_batchnorm=config.getboolean('architecture', 'is-batchnorm'),
            in_channels=1,
            is_deconv=config.getboolean('architecture', 'is-deconv')
        )
        self.interpolate = interpolate
        self.feature_fusion = feature_fusion

        # number of channels + dropout-rate (one per layer)
        # standard-values:
        #   dropout=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        #   channels=[32,64,128,256,512]
        self.channels = [int(i) for i in (config.get('architecture', 'channels')).split(',')]
        self.dropout = [float(i) for i in (config.get('architecture', 'dropout')).split(',')]
        self.model_name = config.get('architecture', 'architecture-name')
        assert len(self.channels)==5
        assert len(self.dropout)==9

        print('Channel-variable: ' + str(self.channels))

        self.conv1 = self._make_layer_2plus3(
            self.in_channels,
            self.channels[0],
            is_batchnorm = self.is_batchnorm,
            is_residual=True,
            dropout = self.dropout[0]
        )
        self.conv2 = self._make_layer_2plus3(
            self.channels[0],
            self.channels[1],
            is_batchnorm = self.is_batchnorm,
            is_residual=True,
            dropout = self.dropout[1]
        )
        self.conv3 = self._make_layer_2plus3(
            self.channels[1],
            self.channels[2],
            is_batchnorm = self.is_batchnorm,
            is_residual=True,
            dropout = self.dropout[2]
        )
        self.conv4 = self._make_layer_2plus3(
            self.channels[2],
            self.channels[3],
            is_batchnorm = self.is_batchnorm,
            is_residual=True,
            dropout = self.dropout[3]
        )
        self.conv5 = self._make_layer_2plus3(
            self.channels[3],
            self.channels[4],
            is_batchnorm = self.is_batchnorm,
            is_residual=True,
            dropout = self.dropout[4]
        )

        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2))
        self.pool4 = nn.MaxPool3d(kernel_size=(2,2,2))

        self.zdimRed1 = self._make_zdimReductionConvPlusFully(
            channels_in=self.channels[0],
            channels_out=self.channels[0],
            num_convreductions=4,
            final_kernelsize=4,
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=0.0
        )
        self.zdimRed2 = self._make_zdimReductionConvPlusFully(
            channels_in=self.channels[1],
            channels_out=self.channels[1],
            num_convreductions=3,
            final_kernelsize=4,
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=0.0
        )
        self.zdimRed3 = self._make_zdimReductionConvPlusFully(
            channels_in=self.channels[2],
            channels_out=self.channels[2],
            num_convreductions=2,
            final_kernelsize=4,
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=0.0
        )
        self.zdimRed4 = self._make_zdimReductionConvPlusFully(
            channels_in=self.channels[3],
            channels_out=self.channels[3],
            num_convreductions=1,
            final_kernelsize=4,
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=0.0
        )
        self.zdimRed5 = self._make_zdimReductionConvPlusFully(
            channels_in=self.channels[4],
            channels_out=self.channels[4],
            num_convreductions=0,
            final_kernelsize=4,
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=0.0
        )

        #--- 2D encoder
        self.conv1_2d = self._make_layer_2plus3_2d(
            1,
            self.channels[0],
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=self.dropout[0]
        )
        self.conv2_2d = self._make_layer_2plus3_2d(
            self.channels[0],
            self.channels[1],
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=self.dropout[1]
        )
        self.conv3_2d = self._make_layer_2plus3_2d(
            self.channels[1],
            self.channels[2],
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=self.dropout[2]
        )
        self.conv4_2d = self._make_layer_2plus3_2d(
            self.channels[2],
            self.channels[3],
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=self.dropout[3]
        )

        self.pool1_2d = nn.MaxPool2d(kernel_size=(1,2))
        self.pool2_2d = nn.MaxPool2d(kernel_size=(1,2))
        self.pool3_2d = nn.MaxPool2d(kernel_size=(2,2))
        self.pool4_2d = nn.MaxPool2d(kernel_size=(2,2))


        if self.feature_fusion == 'concat':
            self.upsampling_module = unet3dUp2modified
        elif self.feature_fusion == 'add':
            self.upsampling_module = unet3dUp2modifiedAdd
        else:
            raise ValueError(
                'Unknown feature_fusion parameter: {}'
                .format(self.feature_fusion)
            )

        # UPsampling:
        self.up_concat4 = self.upsampling_module(
            self.channels[4],
            self.channels[3],
            upfactor=(2,2,1),
            is_deconv=self.is_deconv,
            is_residual=True,
            dropout=self.dropout[5],
            is_batchnorm=self.is_batchnorm
        )
        self.up_concat3 = self.upsampling_module(
            self.channels[3],
            self.channels[2],
            upfactor=(2,2,1),
            is_deconv=self.is_deconv,
            is_residual=True,
            dropout=self.dropout[6],
            is_batchnorm=self.is_batchnorm
        )
        self.up_concat2 = self.upsampling_module(
            self.channels[2],
            self.channels[1],
            upfactor=(1,2,1),
            is_deconv=self.is_deconv,
            is_residual=True,
            dropout=self.dropout[7],
            is_batchnorm=self.is_batchnorm
        )
        self.up_concat1 = self.upsampling_module(
            self.channels[1],
            self.channels[0],
            upfactor=(1,2,1),
            is_deconv=self.is_deconv,
            is_residual=True,
            dropout=self.dropout[8],
            is_batchnorm=self.is_batchnorm
        )

        # final conv to reduce to one channel:
        self.final1 = nn.Conv3d(in_channels=self.channels[0], out_channels=self.n_classes, kernel_size=1)


    def _make_layer_2plus3(self, channels_in, channels_out, is_batchnorm, is_residual, dropout):
        layers = []
        if channels_in == channels_out:
            downsample=None
        else:
            downsample = nn.Sequential(
                nn.Conv3d(channels_in, channels_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm3d(channels_out))
        # two 1x3x3 kernels working within B-scans:
        layers.append(unet3dConvX(
            channels_in,
            channels_out,
            kernel_size=[(1,3,3),(1,3,3)],
            stride=[(1,1,1),(1,1,1)],
            padding=[(0,1,1),(0,1,1)],
            is_batchnorm=is_batchnorm,
            is_residual=is_residual,
            dropout=dropout,
            downsample=downsample
        ))
        # two 1x3x3 kernels working within B-scans + one 3x1x1 kernel working across B-scans:
        layers.append(unet3dConvX(
            channels_out,
            channels_out,
            kernel_size=[(1,3,3),(1,3,3),(3,1,1)],
            stride=[(1,1,1),(1,1,1),(1,1,1)],
            padding=[(0,1,1),(0,1,1),(1,0,0)],
            is_batchnorm=is_batchnorm,
            is_residual=is_residual,
            dropout=dropout,
            downsample=None
        ))
        return nn.Sequential(*layers)


    def _make_layer_2plus3_2d(self, channels_in, channels_out, is_batchnorm, is_residual, dropout):
        layers = []
        if channels_in == channels_out:
            downsample=None
        else:
            downsample = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(channels_out))
        # two 1x3x3 kernels working within B-scans:
        layers.append(unet2dConvX(
            channels_in,
            channels_out,
            kernel_size=[(1,3),(1,3)],
            stride=[(1,1),(1,1)],
            padding=[(0,1),(0,1)],
            is_batchnorm=is_batchnorm,
            is_residual=is_residual,
            dropout=dropout,
            downsample=downsample
        ))
        # two 1x3x3 kernels working within B-scans + one 3x1x1 kernel working across B-scans:
        layers.append(unet2dConvX(
            channels_out,
            channels_out,
            kernel_size=[(1,3),(1,3),(3,1)],
            stride=[(1,1),(1,1),(1,1)],
            padding=[(0,1),(0,1),(1,0)],
            is_batchnorm=is_batchnorm,
            is_residual=is_residual,
            dropout=dropout,
            downsample=None
        ))
        return nn.Sequential(*layers)


    def _make_zdimReductionConvPlusFully(
        self,
        channels_in,
        channels_out,
        num_convreductions,
        final_kernelsize,
        is_batchnorm,
        is_residual,
        dropout
    ):
        layers=[]
        kernel_size=[]
        stride=[]
        padding=[]
        for _i in range(0,num_convreductions):
            kernel_size.append((1,1,3))
            stride.append((1,1,2))
            padding.append((0,0,1))

        if (channels_in != channels_out) or (num_convreductions>0 and is_residual):
            if is_batchnorm:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        channels_in,
                        channels_out,
                        kernel_size=(1,1,1),
                        stride=(1,1,2**(num_convreductions)),
                        bias=False
                    ),
                    nn.BatchNorm3d(channels_out)
                )
            else:
                downsample = nn.Conv3d(
                    channels_in,
                    channels_out,
                    kernel_size=(1,1,1),
                    stride=(1,1,2**(num_convreductions)),
                    bias=True
                )
        else:
            downsample=None

        if num_convreductions>0:
            # X 1x1x3 kernels reducing the dimensionality of z:
            layers.append(unet3dConvX(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                is_batchnorm=is_batchnorm,
                is_residual=is_residual,
                dropout=dropout,
                downsample=downsample
            ))
            # 1x1xN kernel reducing z-dimension to one:
            layers.append(unet3dConvX(
                channels_out,
                channels_out,
                kernel_size=[(1,1,final_kernelsize)],
                stride=[(1,1,1)],
                padding=[(0,0,0)],
                is_batchnorm=is_batchnorm,
                is_residual=False,
                dropout=dropout,
                downsample=None
            ))
        else:
            # 1x1xN kernel reducing z-dimension to one:
            layers.append(unet3dConvX(
                channels_in,
                channels_out,
                kernel_size=[(1,1,final_kernelsize)],
                stride=[(1,1,1)],
                padding=[(0,0,0)],
                is_batchnorm=is_batchnorm,
                is_residual=False,
                dropout=dropout,
                downsample=None
            ))

        return nn.Sequential(*layers)


    def forward(self, oct, slo):
        #--- 2D path downsampling
        conv1_2d = self.conv1_2d(slo)
        pool1_2d = self.pool1_2d(conv1_2d)
        conv2_2d = self.conv2_2d(pool1_2d)
        pool2_2d = self.pool2_2d(conv2_2d)
        conv3_2d = self.conv3_2d(pool2_2d)
        pool3_2d = self.pool3_2d(conv3_2d)
        conv4_2d = self.conv4_2d(pool3_2d)

        #--- 3D path downsampling
        conv1 = self.conv1(oct)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)

        #--- Skip connections from 3D
        # zDim-reduction in between:
        conv1 = self.zdimRed1(conv1)
        conv1 = torch.mean(conv1, dim=4, keepdim=True)
        conv2 = self.zdimRed2(conv2)
        conv2 = torch.mean(conv2, dim=4, keepdim=True)
        conv3 = self.zdimRed3(conv3)
        conv3 = torch.mean(conv3, dim=4, keepdim=True)
        conv4 = self.zdimRed4(conv4)
        conv4 = torch.mean(conv4, dim=4, keepdim=True)
        conv5 = self.zdimRed5(conv5)
        conv5 = torch.mean(conv5, dim=4, keepdim=True)

        #--- Skip connections from 2D
        # Convert from 2D format to 2D within 3D format
        # Example:
        #   from torch.Size([1, 128, 16, 32])
        #   to torch.Size([1, 128, 16, 32, 1])
        #   -> Add 5th dimension
        conv1_2d = conv1_2d[:,:,:,:,None]
        conv2_2d = conv2_2d[:,:,:,:,None]
        conv3_2d = conv3_2d[:,:,:,:,None]
        # size: torch.Size([1, 64, 736, 248, 1])
        #  - Batch, Channels, EF-height, EF-width, depth
        conv4_2d = conv4_2d[:,:,:,:,None]

        if self.interpolate == '2d':
            # From 1, excluding batch and channel dimensions
            conv1_2d = F.interpolate(
                conv1_2d, size=conv1.shape[2:], mode='trilinear',
            )
            conv2_2d = F.interpolate(
                conv2_2d, size=conv2.shape[2:], mode='trilinear',
            )
            conv3_2d = F.interpolate(
                conv3_2d, size=conv3.shape[2:], mode='trilinear',
            )
            conv4_2d = F.interpolate(
                conv4_2d, size=conv4.shape[2:], mode='trilinear',
            )
        elif self.interpolate == '2d_max':
            conv1_2d = F.adaptive_max_pool3d(
                conv1_2d, output_size=conv1.shape[2:]
            )
            conv2_2d = F.adaptive_max_pool3d(
                conv2_2d, output_size=conv2.shape[2:]
            )
            conv3_2d = F.adaptive_max_pool3d(
                conv3_2d, output_size=conv3.shape[2:]
            )
            conv4_2d = F.adaptive_max_pool3d(
                conv4_2d, output_size=conv4.shape[2:]
            )

        #--- 2D within 3D upsampling
        # Each up_concat does the following:
        #   Params: features from encoder, features from deeper level
        #   1. Upsamples 2nd element
        #   2. Concatenates both elements
        #   3. Performs convolution
        up4 = self.up_concat4(conv4, conv4_2d, conv5)
        up3 = self.up_concat3(conv3, conv3_2d, up4)
        up2 = self.up_concat2(conv2, conv2_2d, up3)
        up1 = self.up_concat1(conv1, conv1_2d, up2)

        # get the 'segmentation'
        final_out = self.final1(up1)

        return final_out



class ModifiedUnet3D2DLevel5(ModifiedUnet3D2D):
    '''Convolution-Block with X convolutions in 3D.'''
    def __init__(
        self,
        config,
        interpolate: Union[str, None] = None,
        feature_fusion: str = 'concat'
    ):
        super().__init__(config, interpolate, feature_fusion)
        self.conv5_2d = self._make_layer_2plus3_2d(
            self.channels[3],
            self.channels[4],
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=self.dropout[4]
        )
        self.up_concat4 = self.upsampling_module(
            self.channels[4]*2,
            self.channels[3],
            upfactor=(2,2,1),
            is_deconv=self.is_deconv,
            is_residual=True,
            dropout=self.dropout[5],
            is_batchnorm=self.is_batchnorm
        )

    def forward(self, oct, slo):
        # Example: after-zdimRed4-conv4.shape: torch.Size([1, 128, 16, 32, 1])

        #--- 2D path downsampling
        conv1_2d = self.conv1_2d(slo)
        pool1_2d = self.pool1_2d(conv1_2d)
        conv2_2d = self.conv2_2d(pool1_2d)
        pool2_2d = self.pool2_2d(conv2_2d)
        conv3_2d = self.conv3_2d(pool2_2d)
        pool3_2d = self.pool3_2d(conv3_2d)
        conv4_2d = self.conv4_2d(pool3_2d)
        pool4_2d = self.pool4_2d(conv4_2d)
        conv5_2d = self.conv5_2d(pool4_2d)

        #--- 3D path downsampling
        conv1 = self.conv1(oct)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)

        #--- Skip connections from 3D
        # NOTE: original has no means
        # zDim-reduction in between:
        conv1 = self.zdimRed1(conv1)
        conv1 = torch.mean(conv1, dim=4, keepdim=True)
        conv2 = self.zdimRed2(conv2)
        conv2 = torch.mean(conv2, dim=4, keepdim=True)
        conv3 = self.zdimRed3(conv3)
        conv3 = torch.mean(conv3, dim=4, keepdim=True)
        conv4 = self.zdimRed4(conv4)
        conv4 = torch.mean(conv4, dim=4, keepdim=True)
        conv5 = self.zdimRed5(conv5)
        conv5 = torch.mean(conv5, dim=4, keepdim=True)

        #--- Skip connections from 2D
        # Convert from 2D format to 2D within 3D format
        # Example:
        #   from torch.Size([1, 128, 16, 32])
        #   to torch.Size([1, 128, 16, 32, 1])
        #   -> Add 5th dimension
        conv1_2d = conv1_2d[:,:,:,:,None]
        conv2_2d = conv2_2d[:,:,:,:,None]
        conv3_2d = conv3_2d[:,:,:,:,None]
        # size: torch.Size([1, 64, 736, 248, 1])
        #  - Batch, Channels, EF-height, EF-width, depth
        conv4_2d = conv4_2d[:,:,:,:,None]
        conv5_2d = conv5_2d[:,:,:,:,None]

        if self.interpolate == '2d':
            # From 1, excluding batch and channel dimensions
            conv1_2d = F.interpolate(conv1_2d, size=conv1.shape[2:], mode='trilinear')
            conv2_2d = F.interpolate(conv2_2d, size=conv2.shape[2:], mode='trilinear')
            conv3_2d = F.interpolate(conv3_2d, size=conv3.shape[2:], mode='trilinear')
            conv4_2d = F.interpolate(conv4_2d, size=conv4.shape[2:], mode='trilinear')
            conv5_2d = F.interpolate(conv5_2d, size=conv5.shape[2:], mode='trilinear')
        elif self.interpolate == '2d_max':
            conv1_2d = F.adaptive_max_pool3d(conv1_2d, output_size=conv1.shape[2:])
            conv2_2d = F.adaptive_max_pool3d(conv2_2d, output_size=conv2.shape[2:])
            conv3_2d = F.adaptive_max_pool3d(conv3_2d, output_size=conv3.shape[2:])
            conv4_2d = F.adaptive_max_pool3d(conv4_2d, output_size=conv4.shape[2:])
            conv5_2d = F.adaptive_max_pool3d(conv5_2d, output_size=conv5.shape[2:])

        #--- 2D within 3D upsampling
        # Each up_concat does the following:
        #   Params: features from encoder, features from deeper level
        #   1. Upsamples 2nd element
        #   2. Concatenates both elements
        #   3. Performs convolution
        conv5 = torch.cat([conv5, conv5_2d], 1)
        up4 = self.up_concat4(conv4, conv4_2d, conv5)
        up3 = self.up_concat3(conv3, conv3_2d, up4)
        up2 = self.up_concat2(conv2, conv2_2d, up3)
        up1 = self.up_concat1(conv1, conv1_2d, up2)

        # get the 'segmentation'
        final_out = self.final1(up1)

        return final_out



class unet3dConvX(nn.Module):
    '''Convolution-Block with X convolutions in 3D.
    Convolutional block:
        1.path: [X-1 times [Conv3d - Batch normalization - Relu] ] + [Conv3d - Batch normalization]
        2.path: identity
        then ReLU
    '''

    def __init__(self, in_size, out_size, kernel_size, stride, padding, is_batchnorm, is_residual, dropout, downsample):
        super(unet3dConvX, self).__init__()

        layers = []
        for i in range(0,len(kernel_size)):
            if is_batchnorm:
                if i==0 and i<len(kernel_size)-1:
                    layers.append(nn.Sequential(
                        nn.Conv3d(
                            in_channels=in_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[i],
                            stride=stride[i],
                            padding=padding[i],
                            bias=not(is_batchnorm)
                        ),
                        nn.BatchNorm3d(out_size),
                        nn.ReLU()
                    ))
                elif i==0 and i==len(kernel_size)-1:
                    layers.append(nn.Sequential(
                        nn.Conv3d(
                            in_channels=in_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[len(kernel_size)-1],
                            stride=stride[len(kernel_size)-1],
                            padding=padding[len(kernel_size)-1],
                            bias=not(is_batchnorm)
                        ),
                        nn.BatchNorm3d(out_size)
                    ))
                elif 0<i<len(kernel_size)-1:
                    layers.append(nn.Sequential(
                        nn.Conv3d(
                            in_channels=out_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[i],
                            stride=stride[i],
                            padding=padding[i],
                            bias=not(is_batchnorm)
                        ),
                        nn.BatchNorm3d(out_size),
                        nn.ReLU()
                    ))
                elif i>0 and i==len(kernel_size)-1:
                    layers.append(nn.Sequential(
                        nn.Conv3d(
                            in_channels=out_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[len(kernel_size)-1],
                            stride=stride[len(kernel_size)-1],
                            padding=padding[len(kernel_size)-1],
                            bias=not(is_batchnorm)
                        ),
                        nn.BatchNorm3d(out_size)
                    ))
                else:
                    raise AssertionError(
                        'UserException: in module "unet3dConvX".'
                        ' Error when value of iterator is "' + str(i)
                    )

            else:
                if i==0 and i<len(kernel_size)-1:
                    layers.append(nn.Sequential(
                        nn.Conv3d(
                            in_channels=in_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[i],
                            stride=stride[i],
                            padding=padding[i],
                            bias=not(is_batchnorm)
                        ),
                        nn.ReLU()
                    ))
                elif i==0 and i==len(kernel_size)-1:
                    layers.append(nn.Conv3d(
                        in_channels=in_size,
                        out_channels=out_size,
                        kernel_size=kernel_size[len(kernel_size)-1],
                        stride=stride[len(kernel_size)-1],
                        padding=padding[len(kernel_size)-1],
                        bias=not(is_batchnorm)
                    ))
                elif 0<i<len(kernel_size)-1:
                    layers.append(nn.Sequential(
                        nn.Conv3d(
                            in_channels=out_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[i],
                            stride=stride[i],
                            padding=padding[i],
                            bias=not(is_batchnorm)
                        ),
                        nn.ReLU()
                    ))
                elif i>0 and i==len(kernel_size)-1:
                    layers.append(nn.Conv3d(
                        in_channels=out_size,
                        out_channels=out_size,
                        kernel_size=kernel_size[len(kernel_size)-1],
                        stride=stride[len(kernel_size)-1],
                        padding=padding[len(kernel_size)-1],
                        bias=not(is_batchnorm)
                    ))
                else:
                    raise AssertionError(
                        'UserException: in module "unet3dConvX".'
                        ' Error when value of iterator is "' + str(i)
                    )


        self.convBlock = nn.Sequential(*layers)

        self.is_residual = is_residual
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None


    def forward(self, x):
        residual = x

        out = self.convBlock(x)
        if self.downsample is not None:
            residual = self.downsample(x)

        if self.is_residual:
            out += residual

        out = self.relu(out)

        if not (self.drop is None):
            out = self.drop(out)

        return out


class unet2dConvX(nn.Module):
    '''Convolution-Block with X convolutions in 2D.
    Convolutional block:
        1.path: [X-1 times [Conv2d - Batch normalization - Relu] ] + [Conv2d - Batch normalization]
        2.path: identity
        then ReLU
    '''

    def __init__(
        self,
        in_size,
        out_size,
        kernel_size,
        stride,
        padding,
        is_batchnorm,
        is_residual,
        dropout,
        downsample
    ):
        super().__init__()

        layers = []
        for i in range(0,len(kernel_size)):
            if is_batchnorm:
                if i==0 and i<len(kernel_size)-1:
                    layers.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[i],
                            stride=stride[i],
                            padding=padding[i],
                            bias=not(is_batchnorm)
                        ),
                        nn.BatchNorm2d(out_size),
                        nn.ReLU()
                    ))
                elif i==0 and i==len(kernel_size)-1:
                    layers.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[len(kernel_size)-1],
                            stride=stride[len(kernel_size)-1],
                            padding=padding[len(kernel_size)-1],
                            bias=not(is_batchnorm)
                        ),
                        nn.BatchNorm2d(out_size)
                    ))
                elif 0<i<len(kernel_size)-1:
                    layers.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels=out_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[i],
                            stride=stride[i],
                            padding=padding[i],
                            bias=not(is_batchnorm)
                        ),
                        nn.BatchNorm2d(out_size),
                        nn.ReLU()
                    ))
                elif i>0 and i==len(kernel_size)-1:
                    layers.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels=out_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[len(kernel_size)-1],
                            stride=stride[len(kernel_size)-1],
                            padding=padding[len(kernel_size)-1],
                            bias=not(is_batchnorm)
                        ),
                        nn.BatchNorm2d(out_size)
                    ))
                else:
                    raise AssertionError(
                        'UserException: in module "unet2dConvX".'
                        ' Error when value of iterator is "' + str(i)
                    )

            else:
                if i==0 and i<len(kernel_size)-1:
                    layers.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[i],
                            stride=stride[i],
                            padding=padding[i],
                            bias=not(is_batchnorm)
                        ),
                        nn.ReLU()
                    ))
                elif i==0 and i==len(kernel_size)-1:
                    layers.append(nn.Conv2d(
                        in_channels=in_size,
                        out_channels=out_size,
                        kernel_size=kernel_size[len(kernel_size)-1],
                        stride=stride[len(kernel_size)-1],
                        padding=padding[len(kernel_size)-1],
                        bias=not(is_batchnorm)
                    ))
                elif 0<i<len(kernel_size)-1:
                    layers.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels=out_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[i],
                            stride=stride[i],
                            padding=padding[i],
                            bias=not(is_batchnorm)
                        ),
                        nn.ReLU()
                    ))
                elif i>0 and i==len(kernel_size)-1:
                    layers.append(nn.Conv2d(
                        in_channels=out_size,
                        out_channels=out_size,
                        kernel_size=kernel_size[len(kernel_size)-1],
                        stride=stride[len(kernel_size)-1],
                        padding=padding[len(kernel_size)-1],
                        bias=not(is_batchnorm)
                    ))
                else:
                    raise AssertionError(
                        'UserException: in module "unet2dConvX".'
                        ' Error when value of iterator is "' + str(i)
                    )


        self.convBlock = nn.Sequential(*layers)

        self.is_residual = is_residual
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None


    def forward(self, x):
        residual = x

        out = self.convBlock(x)
        if self.downsample is not None:
            residual = self.downsample(x)

        if self.is_residual:
            out += residual

        out = self.relu(out)

        if not (self.drop is None):
            out = self.drop(out)

        return out



class unet3dUp2modified(nn.Module):
    '''ModifiedUpsampling-Block 3D.'''

    def __init__(
        self,
        lowlayer_channels,
        currlayer_channels,
        upfactor,
        is_deconv,
        is_residual,
        dropout,
        is_batchnorm
    ):
        super(unet3dUp2modified, self).__init__()
        # first an upsampling operation
        if is_deconv:
            self.up = nn.ConvTranspose3d(
                lowlayer_channels,
                currlayer_channels,
                kernel_size=upfactor,
                stride=upfactor
            )
        else:
            self.up = Upsample_Custom3d_nearest(scale_factor=upfactor, mode='nearest')

        # and then a convolution
        if is_batchnorm:
            downsample = nn.Sequential(
                nn.Conv3d(
                    lowlayer_channels+(currlayer_channels * 2),
                    currlayer_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False
                ),
                nn.BatchNorm3d(currlayer_channels)
            )
        else:
            downsample = nn.Conv3d(
                lowlayer_channels+(currlayer_channels * 2),
                currlayer_channels,
                kernel_size=1,
                stride=1,
                bias=True
            )

        self.conv = unet3dConvX(
            in_size=lowlayer_channels + (currlayer_channels * 2),
            out_size=currlayer_channels,
            kernel_size=[(3,3,1),(3,3,1)],
            stride=[(1,1,1),(1,1,1)],
            padding=[(1,1,0),(1,1,0)],
            is_batchnorm=is_batchnorm,
            is_residual=is_residual,
            dropout=dropout,
            downsample=downsample
        )


    def forward(self, inputs1, inputs1_b, inputs2):
        """
        Args:
            inputs1: features from encoder1 (3D)
            inputs1_b: features from encoder2 (2D)
            inputs2: features from deeper level
        """
        # upscale input coming from lower layer
        upsampled_inputs2 = self.up(inputs2)
        # convolution performed on concatenated inputs
        return self.conv(torch.cat([inputs1, inputs1_b, upsampled_inputs2], 1))


class unet3dUp2modifiedAdd(unet3dUp2modified):
    '''ModifiedUpsampling-Block 3D with additive fusion.'''

    def __init__(
        self,
        lowlayer_channels,
        currlayer_channels,
        upfactor,
        is_deconv,
        is_residual,
        dropout,
        is_batchnorm
    ):
        super(unet3dUp2modified, self).__init__()
        # first an upsampling operation
        if is_deconv:
            self.up = nn.ConvTranspose3d(
                lowlayer_channels,
                currlayer_channels,
                kernel_size=upfactor,
                stride=upfactor
            )
        else:
            self.up = Upsample_Custom3d_nearest(scale_factor=upfactor, mode='nearest')

        # and then a convolution
        if is_batchnorm:
            downsample = nn.Sequential(
                nn.Conv3d(
                    lowlayer_channels+currlayer_channels,
                    currlayer_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False
                ),
                nn.BatchNorm3d(currlayer_channels)
            )
        else:
            downsample = nn.Conv3d(
                lowlayer_channels+currlayer_channels,
                currlayer_channels,
                kernel_size=1,
                stride=1,
                bias=True
            )

        self.conv = unet3dConvX(
            in_size=lowlayer_channels + currlayer_channels,
            out_size=currlayer_channels,
            kernel_size=[(3,3,1),(3,3,1)],
            stride=[(1,1,1),(1,1,1)],
            padding=[(1,1,0),(1,1,0)],
            is_batchnorm=is_batchnorm,
            is_residual=is_residual,
            dropout=dropout,
            downsample=downsample
        )


    def forward(self, inputs1, inputs1_b, inputs2):
        """
        Args:
            inputs1: features from encoder1 (3D)
            inputs1_b: features from encoder2 (2D)
            inputs2: features from deeper level
        """
        # upscale input coming from lower layer
        upsampled_inputs2 = self.up(inputs2)
        inputs = inputs1 + inputs1_b
        # convolution performed on concatenated inputs
        return self.conv(torch.cat([inputs, upsampled_inputs2], 1))


