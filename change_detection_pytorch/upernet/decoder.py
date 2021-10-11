import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import Decoder, modules


class PSPBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            modules.Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            PSPBlock(in_channels, out_channels, size, use_bathcnorm=use_bathcnorm) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Sequential(
                nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(pyramid_channels),   # adjust to "SynchronizedBatchNorm2d" if you need.
                nn.ReLU(inplace=True)
            )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
        }[norm]
    return norm(out_channels)


class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat


# class FeatureAlign_V2(nn.Module):  # FaPN full version
#     def __init__(self, in_nc=128, out_nc=128, norm=None):
#         super(FeatureAlign_V2, self).__init__()
#         self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
#         self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
#         self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
#                                 extra_offset_mask=True)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, feat_l, feat_s, main_path=None):
#         HW = feat_l.size()[2:]
#         if feat_l.size()[2:] != feat_s.size()[2:]:
#             feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
#         else:
#             feat_up = feat_s
#         feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
#         offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
#         feat_align = self.relu(self.dcpack_L2([feat_up, offset], main_path))  # [feat, offset]
#         return feat_align + feat_arm


class UPerNetDecoder(Decoder):
    def __init__(
            self,
            encoder_channels,
            encoder_depth=5,
            psp_channels=512,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy="cat",
            fusion_form="concat",
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for UPerNet decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[:encoder_depth + 1]

        # adjust encoder channels according to fusion form
        self.fusion_form = fusion_form
        if self.fusion_form == "concat":
            encoder_channels = [ch*2 for ch in encoder_channels]

        self.psp = PSPModule(
            in_channels=encoder_channels[0],
            out_channels=psp_channels,
            sizes=(1, 2, 3, 6),
            use_bathcnorm=True,
        )

        self.psp_last_conv = modules.Conv2dReLU(
            in_channels=psp_channels * len((1, 2, 3, 6)) + encoder_channels[0],
            out_channels=pyramid_channels,
            kernel_size=1,
            use_batchnorm=True,
        )

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.merge = MergeBlock(merge_policy)

        self.conv_last = modules.Conv2dReLU(self.out_channels, pyramid_channels, 1)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *features):

        features = self.aggregation_layer(features[0], features[1],
                                          self.fusion_form, ignore_original_img=True)
        c2, c3, c4, c5 = features[-4:]

        c5 = self.psp(c5)
        p5 = self.psp_last_conv(c5)

        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        output_size = p2.size()[2:]
        feature_pyramid = [nn.functional.interpolate(p, output_size,
                                                     mode='bilinear', align_corners=False) for p in [p5, p4, p3, p2]]
        x = self.merge(feature_pyramid)
        x = self.conv_last(x)
        # x = self.dropout(x)

        return x
