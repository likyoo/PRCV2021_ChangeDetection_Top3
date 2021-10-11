import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class CBAMChannel(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAMChannel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class CBAMSpatial(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(CBAMSpatial, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """
    Woo S, Park J, Lee J Y, et al. Cbam: Convolutional block attention module[C]
    //Proceedings of the European conference on computer vision (ECCV).
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelGate = CBAMChannel(in_channels, reduction)
        self.SpatialGate = CBAMSpatial(kernel_size)

    def forward(self, x):
        x = self.ChannelGate(x)
        x = self.SpatialGate(x)
        return x


class ECA(nn.Module):
    """ECA-Net: Efficient Channel Attention.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, in_channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ECAM(nn.Module):
    """
    Ensemble Channel Attention Module for UNetPlusPlus.
    Fang S, Li K, Shao J, et al. SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images[J].
    IEEE Geoscience and Remote Sensing Letters, 2021.

    Not completely consistent, to be improved.
    """
    def __init__(self, in_channels, out_channels, map_num=4):
        super(ECAM, self).__init__()
        self.ca1 = CBAMChannel(in_channels * map_num, reduction=16)
        self.ca2 = CBAMChannel(in_channels, reduction=16 // 4)
        self.up = nn.ConvTranspose2d(in_channels * map_num, in_channels * map_num, 2, stride=2)
        self.conv_final = nn.Conv2d(in_channels * map_num, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x (list[tensor] or tuple(tensor))
        """
        out = torch.cat(x, 1)
        intra = torch.sum(torch.stack(x), dim=0)
        ca2 = self.ca2(intra)
        out = self.ca1(out) * (out + ca2.repeat(1, 4, 1, 1))
        out = self.up(out)
        out = self.conv_final(out)
        return out


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return nn.BatchNorm2d


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


class OCR(nn.Module):
    """
    Segmentation Transformer: Object-Contextual Representations for Semantic Segmentation
    https://arxiv.org/pdf/1909.11065.pdf
    """
    def __init__(self, in_channels, num_classes, ocr_mid_channels=512, ocr_key_channels=256):

        super().__init__()
        pre_stage_channels = in_channels
        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(last_inp_channels, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_inp_channels, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):

        out_aux_seg = []

        # ocr
        out_aux = self.aux_head(x)
        # compute contrast feature
        feats = self.conv3x3_ocr(x)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)

        return out_aux_seg



# from timm.models.layers.conv_bn_act import ConvBnAct
# from timm.models.layers.helpers import make_divisible
#
# def _kernel_valid(k):
#     if isinstance(k, (list, tuple)):
#         for ki in k:
#             return _kernel_valid(ki)
#     assert k >= 3 and k % 2
#
#
# class SelectiveKernelAttn(nn.Module):
#     def __init__(self, channels, num_paths=2, attn_channels=32,
#                  act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
#         """ Selective Kernel Attention Module
#
#         Selective Kernel attention mechanism factored out into its own module.
#
#         """
#         super(SelectiveKernelAttn, self).__init__()
#         self.num_paths = num_paths
#         self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
#         self.bn = norm_layer(attn_channels)
#         self.act = act_layer(inplace=True)
#         self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)
#
#     def forward(self, x):
#         assert x.shape[1] == self.num_paths
#         x = x.sum(1).mean((2, 3), keepdim=True)
#         x = self.fc_reduce(x)
#         x = self.bn(x)
#         x = self.act(x)
#         x = self.fc_select(x)
#         B, C, H, W = x.shape
#         x = x.view(B, self.num_paths, C // self.num_paths, H, W)
#         x = torch.softmax(x, dim=1)
#         return x
#
#
# class SelectiveKernelFusion(nn.Module):
#
#     def __init__(self, in_channels, out_channels=None, kernel_size=None, stride=1, dilation=1, groups=1,
#                  rd_ratio=1./16, rd_channels=None, rd_divisor=8, keep_3x3=True, split_input=False,
#                  drop_block=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None):
#         """ Selective Kernel Convolution Module
#
#         As described in Selective Kernel Networks (https://arxiv.org/abs/1903.06586) with some modifications.
#
#         Largest change is the input split, which divides the input channels across each convolution path, this can
#         be viewed as a grouping of sorts, but the output channel counts expand to the module level value. This keeps
#         the parameter count from ballooning when the convolutions themselves don't have groups, but still provides
#         a noteworthy increase in performance over similar param count models without this attention layer. -Ross W
#
#         Args:
#             in_channels (int):  module input (feature) channel count
#             out_channels (int):  module output (feature) channel count
#             kernel_size (int, list): kernel size for each convolution branch
#             stride (int): stride for convolutions
#             dilation (int): dilation for module as a whole, impacts dilation of each branch
#             groups (int): number of groups for each branch
#             rd_ratio (int, float): reduction factor for attention features
#             keep_3x3 (bool): keep all branch convolution kernels as 3x3, changing larger kernels for dilations
#             split_input (bool): split input channels evenly across each convolution branch, keeps param count lower,
#                 can be viewed as grouping by path, output expands to module out_channels count
#             drop_block (nn.Module): drop block module
#             act_layer (nn.Module): activation layer to use
#             norm_layer (nn.Module): batchnorm/norm layer to use
#         """
#         super(SelectiveKernelFusion, self).__init__()
#         out_channels = out_channels or in_channels
#         kernel_size = kernel_size or [3, 5]  # default to one 3x3 and one 5x5 branch. 5x5 -> 3x3 + dilation
#         _kernel_valid(kernel_size)
#         if not isinstance(kernel_size, list):
#             kernel_size = [kernel_size] * 2
#         if keep_3x3:
#             dilation = [dilation * (k - 1) // 2 for k in kernel_size]
#             kernel_size = [3] * len(kernel_size)
#         else:
#             dilation = [dilation] * len(kernel_size)
#         self.num_paths = len(kernel_size)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.split_input = split_input
#         if self.split_input:
#             assert in_channels % self.num_paths == 0
#             in_channels = in_channels // self.num_paths
#         groups = min(out_channels, groups)
#
#         conv_kwargs = dict(
#             stride=stride, groups=groups, drop_block=drop_block, act_layer=act_layer, norm_layer=norm_layer,
#             aa_layer=aa_layer)
#         self.paths = nn.ModuleList([
#             ConvBnAct(in_channels, out_channels, kernel_size=k, dilation=d, **conv_kwargs)
#             for k, d in zip(kernel_size, dilation)])
#
#         attn_channels = rd_channels or make_divisible(out_channels * rd_ratio, divisor=rd_divisor)
#         self.attn = SelectiveKernelAttn(out_channels, self.num_paths, attn_channels)
#         self.drop_block = drop_block
#
#     def forward(self, x1, x2):
#         if self.split_input:
#             # 未定义
#             x_split = torch.split(x1, self.in_channels // self.num_paths, 1)
#             x_paths = [op(x_split[i]) for i, op in enumerate(self.paths)]
#         else:
#             x_paths = [op(x1) for op in self.paths]
#         x = torch.stack(x_paths, dim=1)
#         x_attn = self.attn(x)
#         x = x * x_attn
#         x = torch.sum(x, dim=1)
#         return x


class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)


class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        elif name == 'cbam_channel':
            self.attention = CBAMChannel(**params)
        elif name == 'cbam_spatial':
            self.attention = CBAMSpatial(**params)
        elif name == 'cbam':
            self.attention = CBAM(**params)
        elif name == 'eca':
            self.attention = ECA(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
