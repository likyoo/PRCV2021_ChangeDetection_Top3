import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import DeformConv2d

from ..base import Decoder


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class DCNv2(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=1):
        super().__init__()
        self.dcn = DeformConv2d(c1, c2, k, s, p, groups=g)
        self.offset_mask = nn.Conv2d(c2, g * 3 * k * k, k, s, p)
        self._init_offset()

    def _init_offset(self):
        self.offset_mask.weight.data.zero_()
        self.offset_mask.bias.data.zero_()

    def forward(self, x, offset):
        out = self.offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = mask.sigmoid()
        return self.dcn(x, offset, mask)


class FSM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv_atten = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
        feat = torch.mul(x, atten)
        x = x + feat
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.lateral_conv = FSM(c1, c2)
        self.offset = nn.Conv2d(c2 * 2, c2, 1, bias=False)
        self.dcpack_l2 = DCNv2(c2, c2, 3, 1, 1, 8)

    def forward(self, feat_l, feat_s):
        feat_up = feat_s
        if feat_l.shape[2:] != feat_s.shape[2:]:
            feat_up = F.interpolate(feat_s, size=feat_l.shape[2:], mode='bilinear', align_corners=False)

        feat_arm = self.lateral_conv(feat_l)
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))

        feat_align = F.relu(self.dcpack_l2(feat_up, offset))
        return feat_align + feat_arm


class FaPNDecoder(Decoder):
    def __init__(
            self,
            encoder_channels,
            encoder_depth=5,
            pyramid_channels=128,
            dropout=0.1,
            fusion_form="concat",
    ):
        super().__init__()

        self.out_channels = pyramid_channels
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[:encoder_depth + 1]

        # adjust encoder channels according to fusion form
        self.fusion_form = fusion_form
        if self.fusion_form == "concat":
            encoder_channels = [ch*2 for ch in encoder_channels]

        self.align_modules = nn.ModuleList([ConvModule(encoder_channels[0], pyramid_channels, 1)])
        self.output_convs = nn.ModuleList([])

        for ch in encoder_channels[1:]:
            self.align_modules.append(FAM(ch, pyramid_channels))
            self.output_convs.append(ConvModule(pyramid_channels, pyramid_channels, 3, 1, 1))

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *features):

        features = self.aggregation_layer(features[0], features[1],
                                          self.fusion_form, ignore_original_img=True)
        features = features[-4:]
        features = features[::-1]

        out = self.align_modules[0](features[0])

        for feat, align_module, output_conv in zip(features[1:], self.align_modules[1:], self.output_convs):
            out = align_module(feat, out)
            out = output_conv(out)

        return out
