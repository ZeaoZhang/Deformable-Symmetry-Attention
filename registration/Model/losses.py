

import math
import torch
import numpy as np
from Model.config import args
import torch.nn.functional as F
from pytorch_msssim import SSIM, MS_SSIM
import pystrum.pynd.ndutils as nd


class Grad(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class mse_loss(torch.nn.Module):
    def __init__(self) -> None:
        super(mse_loss, self).__init__()

    def forward(self, x, y):
        return torch.mean((x - y) ** 2)



def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class ncc(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(ncc, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda:{}".format(args.gpu))

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross


def cc_loss(x, y):
    # Calculations based on the cross-correlation formula
    dim = [2, 3, 4]
    mean_x = torch.mean(x, dim, keepdim=True)
    mean_y = torch.mean(y, dim, keepdim=True)
    mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
    mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
    stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
    stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
    return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))



class nj_loss(torch.nn.Module):
    def __init__(self, func='rstf2d') -> None:
        super(nj_loss, self).__init__()
        assert func in ['rstf2d', 'rstf3d', 'stf']
        self.mode=func


    def get_Ja(self, flow):
        """
        jacobian determinant of a displacement field.
        NB: to compute the spatial gradients, we use np.gradient.
        Parameters:
            disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
                where vol_shape is of len nb_dims
        Returns:
            jacobian determinant (scalar)
        """

        # check inputs

        theta = flow[:, 0, :, :].unsqueeze(1)   # b,1,y,x
        w1 = torch.cat([torch.cos(theta * torch.pi), torch.sin(theta * torch.pi)], dim=1).unsqueeze(2)  # b,2,1,y,x
        w2 = torch.cat([-1.0 * torch.sin(theta * torch.pi), torch.cos(theta * torch.pi)], dim=1).unsqueeze(2)  # b,2,1,y,x
        w = torch.cat([w1, w2], dim=2)  # b,2,2,y,x
        volshape = flow.shape[2:]
        nb_dims = len(volshape)
        assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
        grid_lst = nd.volsize2ndgrid(volshape)
        grid = torch.as_tensor(np.stack(grid_lst, len(volshape)), device=flow.device, dtype=flow.dtype).permute(2, 0, 1).unsqueeze(0)

        new_locs = None
        if self.func == 'rstf2d':
            # compute grid
            new_locs = torch.einsum('bijyx,bjyx->biyx', w, flow[:, 1:, :, :]) - flow[:, 1:, :, :] + grid
            new_locs = torch.squeeze(new_locs).permute(1, 2, 0)
            # compute gradients
            
        elif self.func == 'stf':
            new_locs = flow + grid
            
        elif self.func == 'resf3d':
            theta = flow[:, 0, :, :, :].unsqueeze(1)   # b,1,c,h,w
            phi = flow[:, 1, :, :, :].unsqueeze(1)   # b,1,c,h,w
            xi = flow[:, 2, :, :, :].unsqueeze(1)   # b,1,c,h,w
            yi = flow[:, 3, :, :, :].unsqueeze(1)   # b,1,c,h,w
            zi = flow[:, 4, :, :, :].unsqueeze(1)   # b,1,c,h,w
            r1 = torch.sqrt(torch.pow(xi, 2) + torch.pow(yi, 2) + torch.pow(zi, 2))  # b,1,c,h,w
            r2 = torch.sqrt(torch.pow(xi, 2) + torch.pow(yi, 2))  
            x = (r2 * torch.cos(theta) + torch.sin(theta) * zi) * ((xi * torch.cos(phi) - yi * torch.sin(phi)) / r2)
            y = (r2 * torch.cos(theta) + torch.sin(theta) * zi) * ((yi * torch.cos(phi) + xi * torch.sin(phi)) / r2)
            z = zi * torch.cos(theta) - r2 * torch.sin(theta)            
            new_locs = grid + torch.cat([y, x, z]) - flow[:, 2:, ...]  # b,3,c,h,w

        J = torch.gradient(new_locs) # type: ignore

        # 3D glow
        if nb_dims == 3:
            dx = J[0]
            dy = J[1]
            dz = J[2]

            # compute jacobian components
            Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
            Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
            Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

            return Jdet0 - Jdet1 + Jdet2

        else:  # must be 2

            dfdx = J[0]
            dfdy = J[1]

            return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

    def forward(self, flow):
        '''
        Penalizing locations where Jacobian has negative determinants
        '''
        Ja = self.get_Ja(flow)
        Neg_Jac = 0.5 * (torch.abs(Ja) - Ja)
        return torch.sum(Neg_Jac)


class ssim_loss(torch.nn.Module):
    def __init__(self, if_MS=True, win_size=11):
        super(ssim_loss, self).__init__()
        if if_MS:
            self.SSIM = MS_SSIM(win_size=win_size, data_range=255, size_average=True, channel=1)
        else:
            self.SSIM = SSIM(data_range=255, size_average=True, channel=1)
    def forward(self, img1, img2):
        return -self.SSIM(img1, img2)


class Bend_Penalty(torch.nn.Module):
    """
    Bending Penalty of the spatial transformation (2D)
    """
    def __init__(self):
        super(Bend_Penalty, self).__init__()
    
    def _diffs(self, y, dim):#y shape(bs, nfeat, vol_shape)
        ndims = y.ndimension() - 2
        d = dim + 2
        # permute dimensions to put the ith dimension first
#       r = [d, *range(d), *range(d + 1, ndims + 2)]
        y = y.permute(d, *range(d), *range(d + 1, ndims + 2))
        dfi = y[1:, ...] - y[:-1, ...]
        
        # permute back
        # note: this might not be necessary for this loss specifically,
        # since the results are just summed over anyway.
#       r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
        df = dfi.permute(*range(1, d + 1), 0, *range(d + 1, ndims + 2))
        
        return df
    
    def forward(self, pred):#shape(B,C,H,W)
        Ty = self._diffs(pred, dim=0)
        Tx = self._diffs(pred, dim=1)
        Tyy = self._diffs(Ty, dim=0)
        Txx = self._diffs(Tx, dim=1)
        Txy = self._diffs(Tx, dim=0)
        p = Tyy.pow(2).mean() + Txx.pow(2).mean() + 2 * Txy.pow(2).mean()
        
        return p
