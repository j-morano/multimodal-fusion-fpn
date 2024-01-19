import torch
import torch.nn as nn
from models.fpn.components import unet3dConvX, unet3dUp2modified, SegmentationNetwork
from config import config as global_config



class ModifiedUnet3D(SegmentationNetwork):
    '''ModifiedUnet3D U-Net architecture. Fully convolutional neural
    network with encoder/decoder architecture and skipping connections.
    '''

    def __init__(
        self,
        config,
        original=False,
        classification=False,
    ):
        super(ModifiedUnet3D, self).__init__(
            n_classes=global_config.number_of_outputs,
            is_batchnorm=config.getboolean('architecture', 'is-batchnorm'),
            in_channels=1,
            is_deconv=config.getboolean('architecture', 'is-deconv'),
        )
        # Z x W x H
        self.use_1x1 = True
        self.original = original
        self.classification = classification

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

        if self.original:
            final_kernel_size = 8
        else:
            final_kernel_size = 4

        self.zdimRed1 = self._make_zdimReductionConvPlusFully(
            channels_in=self.channels[0],
            channels_out=self.channels[0],
            num_convreductions=4,
            final_kernelsize=final_kernel_size,
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=0.0
        )
        self.zdimRed2 = self._make_zdimReductionConvPlusFully(
            channels_in=self.channels[1],
            channels_out=self.channels[1],
            num_convreductions=3,
            final_kernelsize=final_kernel_size,
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=0.0
        )
        self.zdimRed3 = self._make_zdimReductionConvPlusFully(
            channels_in=self.channels[2],
            channels_out=self.channels[2],
            num_convreductions=2,
            final_kernelsize=final_kernel_size,
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=0.0
        )
        self.zdimRed4 = self._make_zdimReductionConvPlusFully(
            channels_in=self.channels[3],
            channels_out=self.channels[3],
            num_convreductions=1,
            final_kernelsize=final_kernel_size,
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=0.0
        )
        self.zdimRed5 = self._make_zdimReductionConvPlusFully(
            channels_in=self.channels[4],
            channels_out=self.channels[4],
            num_convreductions=0,
            final_kernelsize=final_kernel_size,
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=0.0
        )

        # UPsampling:
        self.up_concat4 = unet3dUp2modified(
            self.channels[4],
            self.channels[3],
            upfactor=(2,2,1),
            is_deconv=self.is_deconv,
            is_residual=True,
            dropout=self.dropout[5],
            is_batchnorm=self.is_batchnorm
        )
        self.up_concat3 = unet3dUp2modified(
            self.channels[3],
            self.channels[2],
            upfactor=(2,2,1),
            is_deconv=self.is_deconv,
            is_residual=True,
            dropout=self.dropout[6],
            is_batchnorm=self.is_batchnorm
        )
        self.up_concat2 = unet3dUp2modified(
            self.channels[2],
            self.channels[1],
            upfactor=(1,2,1),
            is_deconv=self.is_deconv,
            is_residual=True,
            dropout=self.dropout[7],
            is_batchnorm=self.is_batchnorm
        )
        self.up_concat1 = unet3dUp2modified(
            self.channels[1],
            self.channels[0],
            upfactor=(1,2,1),
            is_deconv=self.is_deconv,
            is_residual=True,
            dropout=self.dropout[8],
            is_batchnorm=self.is_batchnorm
        )

        # final conv to reduce to one channel:
        self.final1 = nn.Conv3d(
            in_channels=self.channels[0],
            out_channels=self.n_classes,
            kernel_size=1
        )

        if self.classification:
            to_disable = [
                self.zdimRed1, self.zdimRed2, self.zdimRed3, self.zdimRed4,
                self.zdimRed5,
                self.up_concat4, self.up_concat3, self.up_concat2,
                self.up_concat1,
            ]
            # Disable gradient for all weights in the decoder:
            for module in to_disable:
                for param in module.parameters():
                    param.requires_grad = False

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


    def _make_layer_2plus3_dropoutOnlyAtEnd(self, channels_in, channels_out, is_batchnorm, is_residual, dropout):
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
            dropout=0.0,
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


    def _make_layer_2plus2plus1(self, channels_in, channels_out, is_batchnorm, is_residual, dropout):
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
        # two 1x3x3 kernels working within B-scans:
        layers.append(unet3dConvX(
            channels_out,
            channels_out,
            kernel_size=[(1,3,3),(1,3,3)],
            stride=[(1,1,1),(1,1,1)],
            padding=[(0,1,1),(0,1,1)],
            is_batchnorm=is_batchnorm,
            is_residual=is_residual,
            dropout=dropout,
            downsample=None
        ))
        # one 3x1x1 kernel working across B-scans:
        layers.append(unet3dConvX(
            channels_out,
            channels_out,
            kernel_size=[(3,1,1)],
            stride=[(1,1,1)],
            padding=[(1,0,0)],
            is_batchnorm=is_batchnorm,
            is_residual=is_residual,
            dropout=dropout,
            downsample=None
        ))
        return nn.Sequential(*layers)


    def _make_layer_5(self, channels_in, channels_out, is_batchnorm, is_residual, dropout):
        layers = []
        if channels_in == channels_out:
            downsample=None
        else:
            downsample = nn.Sequential(
                nn.Conv3d(channels_in, channels_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm3d(channels_out))
        # four 1x3x3 kernels working within B-scans + one 3x1x1 kernel working across B-scans :
        layers.append(unet3dConvX(
            channels_in,
            channels_out,
            kernel_size=[(1,3,3),(1,3,3),(1,3,3),(1,3,3),(3,1,1)],
            stride=[(1,1,1),(1,1,1),(1,1,1),(1,1,1),(1,1,1)],
            padding=[(0,1,1),(0,1,1),(0,1,1),(0,1,1),(1,0,0)],
            is_batchnorm=is_batchnorm,
            is_residual=is_residual,
            dropout=dropout,
            downsample=downsample
        ))
        return nn.Sequential(*layers)

    def _normal3Dconv_2(self, channels_in, channels_out, is_batchnorm, is_residual, dropout):
        layers = []
        if channels_in == channels_out:
            downsample=None
        else:
            downsample = nn.Sequential(
                nn.Conv3d(
                    channels_in,
                    channels_out,
                    kernel_size=1,
                    stride=1,
                    bias=False
                ),
                nn.BatchNorm3d(channels_out)
            )
        # two 3x3x3 kernels:
        layers.append(unet3dConvX(
            channels_in,
            channels_out,
            kernel_size=[(3,3,3),(3,3,3)],
            stride=[(1,1,1),(1,1,1)],
            padding=[(1,1,1),(1,1,1)],
            is_batchnorm=is_batchnorm,
            is_residual=is_residual,
            dropout=dropout,
            downsample=downsample
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
        for i in range(0,num_convreductions):
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
                downsample = nn.Sequential(nn.Conv3d(
                    channels_in,
                    channels_out,
                    kernel_size=(1,1,1),
                    stride=(1,1,2**(num_convreductions)),
                    bias=True
                ))
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



    def forward(self, x):
        # Downsampling:
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)

        if self.classification:
            return conv5

        # zDim-reduction in between:
        conv1 = self.zdimRed1(conv1)
        if not self.original:
            conv1 = torch.mean(conv1, dim=4, keepdim=True)
        conv2 = self.zdimRed2(conv2)
        if not self.original:
            conv2 = torch.mean(conv2, dim=4, keepdim=True)
        conv3 = self.zdimRed3(conv3)
        if not self.original:
            conv3 = torch.mean(conv3, dim=4, keepdim=True)
        conv4 = self.zdimRed4(conv4)
        if not self.original:
            conv4 = torch.mean(conv4, dim=4, keepdim=True)
        conv5 = self.zdimRed5(conv5)
        if not self.original:
            conv5 = torch.mean(conv5, dim=4, keepdim=True)

        # Upsampling:
        up4 = self.up_concat4(conv4, conv5)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # get the 'segmentation' using a 1x1 convolution
        if self.use_1x1:
            final_out = self.final1(up1)
        else:
            final_out = up1

        return final_out
