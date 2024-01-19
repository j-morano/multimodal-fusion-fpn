import torch
from torch import nn
import numpy as np



class SegmentationNetwork(nn.Module):
    '''Abstract class defining basic initialization.'''
    def __init__(self, n_classes=1, is_batchnorm=True, in_channels=1, is_deconv=False):
        super().__init__()

        # Indicate the number of output classes
        self.n_classes = n_classes
        # Indicate if the network will use deconvolutions or upsamplings (default)
        self.is_deconv = is_deconv
        # Indicate the number of input channels (by default, 1)
        self.in_channels = in_channels
        # Indicate if BN will be used (by default, True)
        self.is_batchnorm = is_batchnorm



class unet3dUp2modified(nn.Module):

    def __init__(self, lowlayer_channels, currlayer_channels, upfactor, is_deconv, is_residual, dropout, is_batchnorm):
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
            downsample = nn.Sequential(nn.Conv3d(
                lowlayer_channels+currlayer_channels,
                currlayer_channels,
                kernel_size=1,
                stride=1,
                bias=True
            ))

        self.conv = unet3dConvX(
            in_size=lowlayer_channels+currlayer_channels,
            out_size=currlayer_channels,
            kernel_size=[(3,3,1),(3,3,1)],
            stride=[(1,1,1),(1,1,1)],
            padding=[(1,1,0),(1,1,0)],
            is_batchnorm=is_batchnorm,
            is_residual=is_residual,
            dropout=dropout,
            downsample=downsample
        )


    def forward(self, inputs1, inputs2):
        # upscale input coming from lower layer
        upsampled_inputs2 = self.up(inputs2)
        # convolution performed on concatenated inputs
        return self.conv(torch.cat([inputs1, upsampled_inputs2], 1))



class unet3dConvX(nn.Module):
    '''
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
                    layers.append(nn.Sequential(nn.Conv3d(
                        in_channels=in_size,
                        out_channels=out_size,
                        kernel_size=kernel_size[len(kernel_size)-1],
                        stride=stride[len(kernel_size)-1],
                        padding=padding[len(kernel_size)-1],
                        bias=not(is_batchnorm)
                    )))
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
                    layers.append(nn.Sequential(nn.Conv3d(
                        in_channels=out_size,
                        out_channels=out_size,
                        kernel_size=kernel_size[len(kernel_size)-1],
                        stride=stride[len(kernel_size)-1],
                        padding=padding[len(kernel_size)-1],
                        bias=not(is_batchnorm)
                    )))
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


class Upsample_Custom3d_nearest(nn.Module):
    """Upsamples a given multi-channel 3D (volumetric) data.

    The input data is assumed to be of the form `minibatch x channels x depth x height x width`.
    Hence, for spatial inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor.

    One can give a :attr:`scale_factor` to calculate the output size.

    Args:
        scale_factor (a tuple of ints, optional): the multiplier for the image height / width / depth

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` where
          :math:`D_{out} = floor(D_{in} * scale_factor[-3])` or `size[-3]`
          :math:`H_{out} = floor(H_{in} * scale_factor[-2])` or `size[-2]`
          :math:`W_{out} = floor(W_{in}  * scale_factor[-1])` or `size[-1]`
    """

    def __init__(self, scale_factor=None, mode='nearest'):
        super(Upsample_Custom3d_nearest, self).__init__()
        self.size = None
        self.scale_factor = scale_factor
        self.mode = mode
        # Asser that needed parameters are set
        assert self.scale_factor is not None, "scale_factor must be set"

    def forward(self, input: torch.Tensor):
        #depth wise interpolation
        depth_idx = (np.ceil(np.asarray(list(range(1, 1 + int(input.shape[-3]*self.scale_factor[-3]))))/self.scale_factor[-3]) - 1).astype(int)  # type: ignore
        # row wise interpolation
        row_idx =  (np.ceil(np.asarray(list(range(1, 1 + int(input.shape[-2]*self.scale_factor[-2]))))/self.scale_factor[-2]) - 1).astype(int)  # type: ignore
        # column wise interpolation
        col_idx = (np.ceil(np.asarray(list(range(1, 1 + int(input.shape[-1]*self.scale_factor[-1]))))/self.scale_factor[-1]) - 1).astype(int)  # type: ignore

        # Create nearest-neighbor upsampled matrix and return it:
        return input[:,:,depth_idx,:,:][:,:,:,row_idx,:][:,:,:,:,col_idx]

    def __repr__(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return self.__class__.__name__ + '(' + info + ')'




class Upsample_Custom2d_nearest(nn.Module):
    """Upsamples a given multi-channel 2D (Image) data.

    The input data is assumed to be of the form `minibatch x channels x height x width`.
    Hence, for spatial inputs, we expect a 4D Tensor.

    The algorithms available for upsampling are nearest neighbor.

    One can give a :attr:`scale_factor` to calculate the output size.

    Args:
        scale_factor (a tuple of ints, optional): the multiplier for the image height / width

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`
          :math:`H_{out} = floor(H_{in} * scale\_factor[-2])` or `size[-2]`
          :math:`W_{out} = floor(W_{in}  * scale\_factor[-1])` or `size[-1]`
    """

    def __init__(self, scale_factor=None, mode='nearest'):
        super(Upsample_Custom2d_nearest, self).__init__()
        self.size = None
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        # row wise interpolation
        row_idx =  (np.ceil(np.asarray(list(range(1, 1 + int(input.shape[-2]*self.scale_factor[-2]))))/self.scale_factor[-2]) - 1).astype(int)  # type: ignore
        # column wise interpolation
        col_idx = (np.ceil(np.asarray(list(range(1, 1 + int(input.shape[-1]*self.scale_factor[-1]))))/self.scale_factor[-1]) - 1).astype(int)  # type: ignore

        # Create nearest-neighbor upsampled matrix and return it:
        return input[:,:,row_idx,:][:,:,:,col_idx]


    def __repr__(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return self.__class__.__name__ + '(' + info + ')'
