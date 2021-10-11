import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md
from ..base import Decoder


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class PSPBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            md.Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            PSPBlock(in_channels, in_channels // len(sizes), size, use_bathcnorm=use_bathcnorm) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        # blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]
        #
        # if n_upsamples > 1:
        #     for _ in range(1, n_upsamples):
        #         blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))
        #
        # self.block = nn.Sequential(*blocks)
        self.n_upsamples = n_upsamples

    def forward(self, x):
        # return self.block(x)
        if self.n_upsamples > 0:
            x = F.interpolate(x, scale_factor=self.n_upsamples * 2, mode="bilinear", align_corners=True)
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


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


# class DecoderBlock(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             skip_channels,
#             out_channels,
#             use_batchnorm=True,
#             attention_type=None,
#     ):
#         super().__init__()
#         self.conv1 = md.Conv2dReLU(
#             in_channels + skip_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
#         self.conv2 = md.Conv2dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.attention2 = md.Attention(attention_type, in_channels=out_channels)
#
#     def forward(self, x, skip=None):
#         x = F.interpolate(x, scale_factor=2, mode="nearest")
#
#         if skip is not None:
#             x = torch.cat([x, skip], dim=1)
#             x = self.attention1(x)
#         x = self.conv1(x)
#         residual = x
#         x = self.conv2(x)
#         x = self.attention2(x)
#
#         x = x + residual
#         return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UFPNDecoder(Decoder):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            fusion_form="concat",
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # adjust encoder channels according to fusion form
        self.fusion_form = fusion_form
        if self.fusion_form == "concat":
            skip_channels = [ch*2 for ch in skip_channels]
            in_channels[0] = in_channels[0] * 2
            head_channels = head_channels * 2

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()
            # self.center = PSPModule(
            #     in_channels=head_channels,
            #     sizes=(1, 2, 3, 6),
            #     use_bathcnorm=use_batchnorm,
            # )
            # in_channels[0] = in_channels[0] + head_channels

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        pyramid_channels = out_channels
        segmentation_channels = 64
        merge_policy = "cat"
        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channel, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples, pyramid_channel in zip([8, 4, 2, 1, 0], pyramid_channels)
        ])

        self.merge = MergeBlock(merge_policy)
        self.final_conv = nn.Conv2d(sum(pyramid_channels), decoder_channels[-1], kernel_size=(1, 1))

        if fusion_form == "conv":
            fusion_layers = [md.Conv2dReLU(in_channel*2, in_channel, kernel_size=3, padding=1) for in_channel in skip_channels[:-1][::-1]]
            fusion_layers = fusion_layers + [md.Conv2dReLU(in_channels[0]*2, in_channels[0], kernel_size=3, padding=1), ]
            self.fusion_layers = nn.ModuleList(fusion_layers)

    def forward(self, *features):

        if self.fusion_form == "conv":
            ignore_original_img = True
            fea_len = len(features[0])
            start_idx = 1 if ignore_original_img else 0

            features = [self.fusion_layers[idx-1](torch.cat([features[0][idx], features[1][idx]], dim=1))
                             for idx in range(start_idx, fea_len)]
        else:
            features = self.aggregation_layer(features[0], features[1],
                                              self.fusion_form, ignore_original_img=True)


        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        feature_pyramid = []
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            # feature_pyramid.append(self.seg_blocks[i](x))
            feature_pyramid.append(x)

        # x = self.merge(feature_pyramid)
        # x = self.final_conv(x)

        return feature_pyramid
