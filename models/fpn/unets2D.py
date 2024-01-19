from torch import nn

from models.fpn.fusion3D2D import ModifiedUnet3D2D
from models.fpn.components import unet3dUp2modified
from config import config as global_config



class ModifiedUnet2D(ModifiedUnet3D2D):
    def __init__(self, config, output_features: bool=False):
        super(ModifiedUnet3D2D, self).__init__(
            n_classes=global_config.number_of_outputs,
            is_batchnorm=config.getboolean('architecture', 'is-batchnorm'),
            in_channels=1,
            is_deconv=config.getboolean('architecture', 'is-deconv')
        )
        # Input is Z x W x H
        self.output_features = output_features

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


        # UPsampling:
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
        # NOTE: original has nn.ReLU at the end
        # It was desigend for regression, not for segmentation
        if not self.output_features:
            self.final1 = nn.Sequential(nn.Conv3d(
                in_channels=self.channels[0],
                out_channels=self.n_classes,
                kernel_size=1
            ))

    def forward(self, input_):  # type: ignore
        #--- 2D path downsampling
        conv1_2d = self.conv1_2d(input_)
        pool1_2d = self.pool1_2d(conv1_2d)
        conv2_2d = self.conv2_2d(pool1_2d)
        pool2_2d = self.pool2_2d(conv2_2d)
        conv3_2d = self.conv3_2d(pool2_2d)
        pool3_2d = self.pool3_2d(conv3_2d)
        conv4_2d = self.conv4_2d(pool3_2d)

        #--- Skip connections from 2D
        # Convert from 2D format to 2D within 3D format
        # Example:
        #   from torch.Size([1, 128, 16, 32])
        #   to torch.Size([1, 128, 16, 32, 1])
        #   -> Add 5th dimension
        conv1_2d = conv1_2d[:,:,:,:,None]
        conv2_2d = conv2_2d[:,:,:,:,None]
        conv3_2d = conv3_2d[:,:,:,:,None]
        conv4_2d = conv4_2d[:,:,:,:,None]

        #--- 2D within 3D upsampling
        # Each up_concat does the following:
        #   Params: features from encoder, features from deeper level
        #   1. Upsamples 2nd element
        #   2. Concatenates both elements
        #   3. Performs convolution
        # up4 = self.up_concat4(conv4_2d, conv5)
        up3 = self.up_concat3(conv3_2d, conv4_2d)
        up2 = self.up_concat2(conv2_2d, up3)
        up1 = self.up_concat1(conv1_2d, up2)

        # get the 'segmentation'
        if self.output_features:
            return up1
        else:
            return self.final1(up1)


class ModifiedUnet2DLevel5(ModifiedUnet2D):
    def __init__(self, config, output_features: bool=False):
        super().__init__(config, output_features)
        self.conv5_2d = self._make_layer_2plus3_2d(
            self.channels[3],
            self.channels[4],
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=self.dropout[4]
        )

        self.pool4_2d = nn.MaxPool2d(kernel_size=(2,2))


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

    def forward(self, input_):
        # Example: after-zdimRed4-conv4.shape: torch.Size([1, 128, 16, 32, 1])

        #--- 2D path downsampling
        conv1_2d = self.conv1_2d(input_)
        pool1_2d = self.pool1_2d(conv1_2d)
        conv2_2d = self.conv2_2d(pool1_2d)
        pool2_2d = self.pool2_2d(conv2_2d)
        conv3_2d = self.conv3_2d(pool2_2d)
        pool3_2d = self.pool3_2d(conv3_2d)
        conv4_2d = self.conv4_2d(pool3_2d)
        pool4_2d = self.pool4_2d(conv4_2d)
        conv5_2d = self.conv5_2d(pool4_2d)

        #--- Skip connections from 2D
        # Convert from 2D format to 2D within 3D format
        # Example:
        #   from torch.Size([1, 128, 16, 32])
        #   to torch.Size([1, 128, 16, 32, 1])
        #   -> Add 5th dimension
        conv1_2d = conv1_2d[:,:,:,:,None]
        conv2_2d = conv2_2d[:,:,:,:,None]
        conv3_2d = conv3_2d[:,:,:,:,None]
        conv4_2d = conv4_2d[:,:,:,:,None]
        conv5_2d = conv5_2d[:,:,:,:,None]

        #--- 2D within 3D upsampling
        # Each up_concat does the following:
        #   Params: features from encoder, features from deeper level
        #   1. Upsamples 2nd element
        #   2. Concatenates both elements
        #   3. Performs convolution
        up4 = self.up_concat4(conv4_2d, conv5_2d)
        up3 = self.up_concat3(conv3_2d, up4)
        up2 = self.up_concat2(conv2_2d, up3)
        up1 = self.up_concat1(conv1_2d, up2)

        # get the 'segmentation'
        if self.output_features:
            return up1
        else:
            return self.final1(up1)

