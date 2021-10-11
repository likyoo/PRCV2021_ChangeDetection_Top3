from ..backbones.hrnet import HRNet

import torch
from torch import nn
import torch.nn.functional as F


def get_backbone(backbone, pretrained):

    if "hrnet" in backbone:
        backbone = HRNet(backbone, pretrained)
    else:
        exit("\nError: BACKBONE \'%s\' is not implemented!\n" % backbone)

    return backbone


class BaseNet(nn.Module):
    def __init__(self, backbone, pretrained):
        super().__init__()
        self.backbone = get_backbone(backbone, pretrained)

    def base_forward(self, x1, x2):
        b, c, h, w = x1.shape

        x1 = self.backbone.base_forward(x1)[-1]
        x2 = self.backbone.base_forward(x2)[-1]

        out = torch.abs(x1 - x2)
        out = self.head(out)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        return out

    def forward(self, x1, x2, TTA=False):
        if not TTA:
            return self.base_forward(x1, x2)
        else:
            out = self.base_forward(x1, x2)
            out = F.softmax(out, dim=1)
            origin_x1 = x1.clone()
            origin_x2 = x2.clone()

            x1 = origin_x1.flip(2)
            x2 = origin_x2.flip(2)
            cur_out = self.base_forward(x1, x2)
            out += F.softmax(cur_out, dim=1).flip(2)

            x1 = origin_x1.flip(3)
            x2 = origin_x2.flip(3)
            cur_out = self.base_forward(x1, x2)
            out += F.softmax(cur_out, dim=1).flip(3)

            x1 = origin_x1.transpose(2, 3).flip(3)
            x2 = origin_x2.transpose(2, 3).flip(3)
            cur_out = self.base_forward(x1, x2)
            out += F.softmax(cur_out, dim=1).flip(3).transpose(2, 3)

            x1 = origin_x1.flip(3).transpose(2, 3)
            x2 = origin_x2.flip(3).transpose(2, 3)
            cur_out = self.base_forward(x1, x2)
            out += F.softmax(cur_out, dim=1).transpose(2, 3).flip(3)

            x1 = origin_x1.flip(2).flip(3)
            x2 = origin_x2.flip(2).flip(3)
            cur_out = self.base_forward(x1, x2)
            out += F.softmax(cur_out, dim=1).flip(3).flip(2)

            out /= 6.0

            return out
