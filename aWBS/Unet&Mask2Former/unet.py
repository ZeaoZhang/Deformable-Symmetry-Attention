# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.utils.typing_utils import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer, build_conv_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmseg.registry import MODELS
from mmseg.models.utils import UpConvBlock, Upsample, resize
from mmseg.models.backbones.unet import UNet, UpConvBlock, BasicConvBlock, InterpConv, ConvModule
from net import SpatialTransformer
import torch.nn.functional as F
from cbam import CBAM
from dataset import stack_batch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class LearnableGaussianConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, sigma=[1.0, 1.0], mean=[0.0, 0.0], loss_weight = 1.0, limit=None):
        super(LearnableGaussianConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.loss_weight = loss_weight
        # limit : 0.2551
        self.limit = limit
        
        # Initialize learnable parameters for each output channel
        self.sigmas = nn.Parameter(torch.tensor([sigma] * out_channels), requires_grad=True)
        self.means = nn.Parameter(torch.tensor([mean] * out_channels), requires_grad=True)

    def gaussian_kernels(self):
        x_coord = torch.arange(self.kernel_size).to(self.sigmas.device)
        y_coord = torch.arange(self.kernel_size).to(self.sigmas.device)
        x_grid, y_grid = torch.meshgrid(x_coord, y_coord, indexing='ij')
        x_grid = x_grid.unsqueeze(0).unsqueeze(0).float()

        mean_x = (self.kernel_size - 1) / 2. + self.means[:, 0].view(-1, 1, 1, 1)
        mean_y = (self.kernel_size - 1) / 2. + self.means[:, 1].view(-1, 1, 1, 1)

        sigma_x = self.sigmas[:, 0].view(-1, 1, 1, 1)
        sigma_y = self.sigmas[:, 1].view(-1, 1, 1, 1)

        gaussian_kernels = (1./(2.*math.pi*sigma_x*sigma_y)) *\
                        torch.exp(-((x_grid - mean_x)**2 / (2*sigma_x**2)
                                    +(y_grid - mean_y)**2 / (2*sigma_y**2)))

        gaussian_kernels = gaussian_kernels / torch.sum(gaussian_kernels, dim=[2, 3], keepdim=True)

        return gaussian_kernels

    def forward(self, x):
        if self.limit != None:
            self.sigmas.data[self.sigmas.data < self.limit] = self.limit
        kernels = self.gaussian_kernels()
        return F.conv2d(x, kernels, stride=1, padding=self.kernel_size//2)
    
    def loss(self):
        return torch.sum(torch.square(self.sigmas)) * self.loss_weight



class LearnableGaussianConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, sigma=[1.0, 1.0, 1.0], mean=[0.0, 0.0, 0.0], loss_weight=1.0, limit=None):
        super(LearnableGaussianConv3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.loss_weight = loss_weight
        # limit: 0.2551
        self.limit = limit

        # Initialize learnable parameters for each output channel
        self.sigmas = nn.Parameter(torch.tensor([sigma] * out_channels), requires_grad=True)
        self.means = nn.Parameter(torch.tensor([mean] * out_channels), requires_grad=True)

    def gaussian_kernels(self):
        x_coord = torch.arange(self.kernel_size).to(self.sigmas.device)
        y_coord = torch.arange(self.kernel_size).to(self.sigmas.device)
        z_coord = torch.arange(self.kernel_size).to(self.sigmas.device)
        x_grid, y_grid, z_grid = torch.meshgrid(x_coord, y_coord, z_coord, indexing='ij')
        x_grid = x_grid.unsqueeze(0).unsqueeze(0).float()

        mean_x = (self.kernel_size - 1) / 2. + self.means[:, 0].view(-1, 1, 1, 1, 1)
        mean_y = (self.kernel_size - 1) / 2. + self.means[:, 1].view(-1, 1, 1, 1, 1)
        mean_z = (self.kernel_size - 1) / 2. + self.means[:, 2].view(-1, 1, 1, 1, 1)

        sigma_x = self.sigmas[:, 0].view(-1, 1, 1, 1, 1)
        sigma_y = self.sigmas[:, 1].view(-1, 1, 1, 1, 1)
        sigma_z = self.sigmas[:, 2].view(-1, 1, 1, 1, 1)

        gaussian_kernels = (1. / (2. * math.pi * sigma_x * sigma_y * sigma_z)) * \
                           torch.exp(-((x_grid - mean_x) ** 2 / (2 * sigma_x ** 2)
                                       + (y_grid - mean_y) ** 2 / (2 * sigma_y ** 2)
                                       + (z_grid - mean_z) ** 2 / (2 * sigma_z ** 2)))

        gaussian_kernels = gaussian_kernels / torch.sum(gaussian_kernels, dim=[2, 3, 4], keepdim=True)

        return gaussian_kernels

    def forward(self, x):
        if self.limit is not None:
            self.sigmas.data[self.sigmas.data < self.limit] = self.limit
        kernels = self.gaussian_kernels()
        return F.conv3d(x, kernels, stride=1, padding=self.kernel_size // 2)


@MODELS.register_module() 
class DSAUNet(UNet):
    def __init__(self,
                 conv_type: Dict,
                 **kwargs):
        super().__init__(**kwargs)
        self.learn_conv = LearnableGaussianConv(**conv_type)
        
    def forward(self, x):
        img = x[:, 0, :, :].unsqueeze(1)
        img_reg = self.learn_conv(x[:, 1, :, :].unsqueeze(1))
        in_features = torch.cat([img, img_reg], dim=1)
        return super().forward(in_features)
    
    def loss(self):
        loss_kernel =  self.learn_conv.loss()
        return {'loss_kernel': loss_kernel}
    
