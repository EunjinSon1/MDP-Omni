import copy
import pdb
import sys
from unittest.mock import inplace

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
import math
from utils.geometry import *
from utils.image import *

class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


autocast = torch.amp.autocast if torch.__version__ >= '1.6.0' else identity_with


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, norm_type='IN', **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=False, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        if norm_type == 'IN':
            self.bn = nn.InstanceNorm2d(out_channels, momentum=bn_momentum) if bn else None
        elif norm_type == 'BN':
            self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu
        self.init_weights(init_method="xavier")

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu:
            y = F.leaky_relu(y, 0.1, inplace=True)
        return y

    def init_weights(self, init_method="xavier"):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, norm, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=False, **kwargs)
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(4, out_channels)
        else:
            self.norm = None
        self.relu = relu
        self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.norm is not None:
            init_bn(self.norm)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, norm, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=False, **kwargs)
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(4, out_channels)
        else:
            self.norm = None
        self.relu = relu
        self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.norm is not None:
            init_bn(self.norm)

class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels, last_layer=True):
        super(CostRegNet, self).__init__()
        self.last_layer = last_layer

        self.conv1 = Conv3d(in_channels, base_channels * 2, norm='gn', stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, norm='gn', padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, norm='gn', stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, norm='gn', padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, norm='gn', stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, norm='gn', padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, norm='gn', stride=2, padding=1, output_padding=1)
        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, norm='gn', stride=2, padding=1, output_padding=1)
        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, norm='gn', stride=2, padding=1, output_padding=1)

        if in_channels != base_channels:
            self.inner = nn.Conv3d(in_channels, base_channels, 1, 1)
        else:
            self.inner = nn.Identity()

        if self.last_layer:
            self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

        self.pe_proj = nn.Conv3d(in_channels * 3, in_channels, 1, 1, bias=False)

    def forward(self, x, position3d):

        x = x + self.pe_proj(PositionEncoding3D(position3d, x.shape[1]))

        conv0 = x
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = self.inner(conv0) + self.conv11(x)
        if self.last_layer:
            x = self.prob(x)
        return x



def indexToInvdepth(disp_values, idx, start_index=0):
    min_invdepth = disp_values[:, 0]
    max_invdepth = disp_values[:, -1]
    num_invdepth = disp_values.shape[1]

    sample_step_invdepth = \
        (max_invdepth - min_invdepth) / (num_invdepth - 1.0)

    return min_invdepth + \
        (idx - start_index) * sample_step_invdepth


def calculate_index_hypothesis_init(B, img_height, img_width, nhypothesis_init, device):

    idx_hypos = torch.linspace(0,191,steps=nhypothesis_init,device=device)
    idx_hypos = idx_hypos.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B,1,img_height,img_width)

    hypo_intervals = idx_hypos[:,1:]-idx_hypos[:,:-1]
    hypo_intervals = torch.cat((hypo_intervals,hypo_intervals[:,-1].unsqueeze(1)),dim=1)

    return idx_hypos.unsqueeze(1), hypo_intervals.unsqueeze(1)


def amvf(x, max_fov, k):
    x = x.clone().clamp_(max=max_fov)
    min_val, max_val = x.amin(), x.amax()

    x = (x - min_val).div_(max_val - min_val)
    norm_weight_s = 1 - 2 / (1 + ((2 - x) / (x + 1e-6)).pow(k))
    return norm_weight_s


def makeSphericalRays(equirect_size: (int, int),
                      phi_deg: float, phi2_deg=-1.0):
    h, w = equirect_size
    ys, xs = torch.meshgrid([torch.arange(0, h, dtype=torch.float64, device='cuda'), 
                               torch.arange(0, w, dtype=torch.float64, device='cuda')], indexing='ij')

    w_2, h_2 = w / 2.0, (h - 1) / 2.0
    xs = (xs - w_2) / w_2 * math.pi + (math.pi / 2.0)
    if phi2_deg > 0.0:
        med = math.radians(sum(phi2_deg - phi_deg) / 2.0)
        med2 = math.radians((phi2_deg + phi_deg) / 2.0)
        ys = (ys - h_2) / h_2 * med2 - med
    else:
        ys = (ys - h_2) / h_2 * math.radians(phi_deg)
    
    X = -torch.cos(ys) * torch.cos(xs)
    Y = torch.sin(ys) # sphere
    # Y = np.sin(ys) / np.cos(ys) # cylinder
    # Y = ys / np.deg2rad(phi_deg) # perspective cylinder
    Z = torch.cos(ys) * torch.sin(xs)
    rays = torch.cat((torch.reshape(X, [1, -1]),
                           torch.reshape(Y, [1, -1]),
                           torch.reshape(Z, [1, -1]))).to(torch.float64)
    return rays

def spherical_warping(feat, ocams, depth_samples, B,H_p, W_p, phi_deg):
    equirect_size = [H_p, W_p]
    num_depth = depth_samples.shape[2]
    depth_min=depth_samples.min()
    depth_max=depth_samples.max()

    with torch.no_grad():
        rays = makeSphericalRays(equirect_size, phi_deg=phi_deg, phi2_deg=-1.0)
        rays = torch.unsqueeze(rays, 0) 
        pts = rays.unsqueeze(2).repeat(B, 1, num_depth, 1) * depth_samples.view(B, 1, num_depth, -1)
        P3d = applyTransform_volume(ocams.rig2cam, pts)

        p2d,theta = ocams.rayToPixel_volume(P3d,out_theta=True)
        theta = theta.reshape([B, num_depth, equirect_size[0], equirect_size[1]])
        grid = pixelToGrid_volume(p2d, equirect_size,(ocams.height, ocams.width))
        grid = torch.clamp(grid, -2, 1).float()
        position3d = P3d.reshape(B, 3, num_depth, H_p, W_p)

    warped_feat = [F.grid_sample(feat, grid[:,d,...],
            align_corners=True) for d in range(0, num_depth)]
    warped_feat = torch.stack(warped_feat, dim=2)

    return warped_feat,position3d,theta 


def PositionEncoding3D(position3d, C, rescale=4.0):
    B, _, D, H, W = position3d.shape
    div_term = torch.exp(torch.arange(0, C, 2).float() * (-math.log(10000.0) / C)).to(position3d.device)
    div_term = div_term[None, :, None] 

    pe_x = torch.zeros((B, C, D * H * W), dtype=torch.float32, device=position3d.device)
    pos_x = position3d[:, 0].reshape(B, 1, D * H * W)
    pe_x[:, 0::2, :] = torch.sin(pos_x * rescale * div_term).reshape(B, -1, D * H * W)
    pe_x[:, 1::2, :] = torch.cos(pos_x * rescale * div_term).reshape(B, -1, D * H * W) 

    pe_y = torch.zeros((B, C, D * H * W), dtype=torch.float32, device=position3d.device)
    pos_y = position3d[:, 1].reshape(B, 1, D * H * W)
    pe_y[:, 0::2, :] = torch.sin(pos_y * rescale * div_term).reshape(B, -1, D * H * W)
    pe_y[:, 1::2, :] = torch.cos(pos_y * rescale * div_term).reshape(B, -1, D * H * W) 

    pe_z = torch.zeros((B, C, D * H * W), dtype=torch.float32, device=position3d.device)
    pos_z = position3d[:, 2].reshape(B, 1, D * H * W)
    pe_z[:, 0::2, :] = torch.sin(pos_z * rescale * div_term).reshape(B, -1, D * H * W)
    pe_z[:, 1::2, :] = torch.cos(pos_z * rescale * div_term).reshape(B, -1, D * H * W) 

    pe = torch.cat([pe_x, pe_y, pe_z], dim=1).reshape(B, C * 3, D, H, W)

    return pe

def proj_cost(features, sigmoid_param, ocams, selected_idxs, B, H_p, W_p, full_disp_hypo, phi_deg):
    features = torch.unbind(features, dim=1)

    selected_disp_hypos = indexToInvdepth(full_disp_hypo, selected_idxs)
    selected_depth_hypos = 1. / selected_disp_hypos 

    warped_volumes = []
    theta_volumes = []
    with autocast(device_type='cuda', enabled=False):
        for idx, feature in enumerate(features): 
            feature = feature.to(torch.float32) 
            warped_volume, position3d,theta = spherical_warping(feature,
                                              ocams[idx],
                                              selected_depth_hypos, 
                                              B,H_p, W_p,
                                              phi_deg)
            warped_volumes.append(warped_volume)
            theta_volumes.append(theta.cuda())

    warped_volumes = torch.stack(warped_volumes, dim=0) 
    theta_volumes = torch.stack(theta_volumes, dim=0)  
    V, B, C, N, H, W = warped_volumes.shape
    theta_volumes = theta_volumes.unsqueeze(2).expand(-1, -1, C, -1, -1, -1) 
    theta_volumes = amvf(theta_volumes, ocams[0].max_theta, sigmoid_param)

    ref = warped_volumes[0] * theta_volumes[0] + warped_volumes[2] * theta_volumes[2]
    tar = warped_volumes[1] * theta_volumes[1] + warped_volumes[3] * theta_volumes[3]

    IS_volume = ref * tar

    return IS_volume, position3d
