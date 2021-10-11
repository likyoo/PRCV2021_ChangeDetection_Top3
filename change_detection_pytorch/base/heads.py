import torch.nn as nn
from .modules import Flatten, Activation, OCR


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1, align_corners=True):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.Upsample(scale_factor=upsampling, mode='bilinear', align_corners=align_corners) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class SegmentationOCRHead(nn.Module):

    def __init__(self, in_channels, out_channels, activation=None, upsampling=1, align_corners=True):
        super().__init__()
        self.ocr_head = OCR(in_channels, out_channels)
        self.upsampling = nn.Upsample(scale_factor=upsampling, mode='bilinear', align_corners=align_corners) if upsampling > 1 else nn.Identity()
        self.activation = Activation(activation)

    def forward(self, x):
        coarse_pre, pre = self.ocr_head(x)
        coarse_pre = self.activation(self.upsampling(coarse_pre))
        pre = self.activation(self.upsampling(pre))
        return [coarse_pre, pre]


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)


class SegmentationCondHead(nn.Module):

    def __init__(self, in_channels, out_channels, activation=None, upsampling=1, align_corners=True):
        super().__init__()
        self.cond_head = CondHead(in_channels, in_channels//2, num_classes=out_channels)
        self.upsampling = nn.Upsample(scale_factor=upsampling, mode='bilinear', align_corners=align_corners) if upsampling > 1 else nn.Identity()
        self.activation = Activation(activation)

    def forward(self, x):
        coarse_pre, pre = self.cond_head(x)
        coarse_pre = self.activation(self.upsampling(coarse_pre))
        pre = self.activation(self.upsampling(pre))
        return [coarse_pre, pre]


import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class CondHead(nn.Module):
    def __init__(self, in_channel: int = 2048, channel: int = 512, num_classes: int = 19):
        super().__init__()
        self.num_classes = num_classes
        self.weight_num = channel * num_classes
        self.bias_num = num_classes

        self.conv = ConvModule(in_channel, channel, 1)
        self.dropout = nn.Dropout2d(0.1)

        self.guidance_project = nn.Conv2d(channel, num_classes, 1)
        self.filter_project = nn.Conv2d(channel * num_classes, self.weight_num + self.bias_num, 1, groups=num_classes)

    def forward(self, features) -> Tensor:
        # x = self.dropout(self.conv(features[-1]))
        x = self.dropout(self.conv(features))
        B, C, H, W = x.shape
        guidance_mask = self.guidance_project(x)
        cond_logit = guidance_mask

        key = x
        value = x
        guidance_mask = guidance_mask.softmax(dim=1).view(*guidance_mask.shape[:2], -1)
        key = key.view(B, C, -1).permute(0, 2, 1)

        cond_filters = torch.matmul(guidance_mask, key)
        cond_filters /= H * W
        cond_filters = cond_filters.view(B, -1, 1, 1)
        cond_filters = self.filter_project(cond_filters)
        cond_filters = cond_filters.view(B, -1)

        weight, bias = torch.split(cond_filters, [self.weight_num, self.bias_num], dim=1)
        weight = weight.reshape(B * self.num_classes, -1, 1, 1)
        bias = bias.reshape(B * self.num_classes)

        value = value.view(-1, H, W).unsqueeze(0)
        seg_logit = F.conv2d(value, weight, bias, 1, 0, groups=B).view(B, self.num_classes, H, W)

        return [cond_logit, seg_logit]