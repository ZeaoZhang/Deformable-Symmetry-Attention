from typing import Sequence, Dict
import torch
from torch import nn
import math
from monai.networks.nets.swin_unetr import SwinUNETR


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

class DSASwinUNETR(SwinUNETR):
    def __init__(self, conv_cfg: Dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.learn_convs = nn.ModuleList([LearnableGaussianConv3D(**conv_cfg)] * 4)


    def forward(self, x_in):
            in_features = []
            for i in range(4, 8):
                in_features.append(x_in[:, i-4, ...].unsqueeze(1))
                reg_features = self.learn_convs[i-4](x_in[:, i, ...].unsqueeze(1))
                in_features.append(reg_features)
            in_features = torch.cat(in_features, dim=1)

            hidden_states_out = self.swinViT(in_features, self.normalize)
            enc0 = self.encoder1(in_features)
            enc1 = self.encoder2(hidden_states_out[0])
            enc2 = self.encoder3(hidden_states_out[1])
            enc3 = self.encoder4(hidden_states_out[2])
            dec4 = self.encoder10(hidden_states_out[4])
            dec3 = self.decoder5(dec4, hidden_states_out[3])
            dec2 = self.decoder4(dec3, enc3)
            dec1 = self.decoder3(dec2, enc2)
            dec0 = self.decoder2(dec1, enc1)
            out = self.decoder1(dec0, enc0)
            logits = self.out(out)
            return logits
    