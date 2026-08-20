# module.basic.py
#
# Author: Changhee Won (changhee.1.won@gmail.com)
#
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.common import *


class Conv2D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, bn=True, relu=True):
        super(Conv2D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.conv = torch.nn.Conv2d(ch_in, ch_out, kernel_size,
                                    stride, padding=pad, dilation=dilation)
        if self.opts.bn:
            self.bn = torch.nn.BatchNorm2d(ch_out)

    def forward(self, x, residual=None):
        x = self.conv(x)
        if self.opts.bn:
            x = self.bn(x)
        if not residual is None:
            x += residual
        if self.opts.relu:
            x = F.relu(x)
        return x


class FeatureLayers(torch.nn.Module):
    def __init__(self, CH=32, use_rgb=False, downsample_twice=False):
        super(FeatureLayers, self).__init__()
        layers = []
        self.use_rgb = use_rgb
        self.downsample_twice = downsample_twice
        in_channel = 3 if use_rgb else 1
        if downsample_twice:
            layers.append(nn.Sequential(Conv2D(in_channel, CH, 5, 2, 2),
                                        Conv2D(CH, CH, 3, 2, 2, dilation=2)))  # conv[1]
        else:
            layers.append(Conv2D(in_channel, CH, 5, 2, 2))  # conv[1]

        layers += [Conv2D(CH, CH, 3, 1, 1) for _ in range(10)]  # conv[2-11]
        for d in range(2, 5):  # conv[12-17]
            layers += [Conv2D(CH, CH, 3, 1, d, dilation=d) for _ in range(2)]
        layers.append(Conv2D(CH, CH, 3, 1, 1, bn=False, relu=False))  # conv[18]
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, im):
        B, V, C, H, W = im.shape
        im = im.reshape(B*V, C, H, W)

        x = self.layers[0](im)
        for i in range(1, 17, 2):
            x_ = self.layers[i](x)
            x = self.layers[i+1](x_, residual=x)
        x = self.layers[17](x)
        return x.reshape(B, V, x.shape[1], x.shape[2], x.shape[3])