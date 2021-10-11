import torch.nn as nn

from . import base
from . import functional as F
from ..base.modules import Activation
import change_detection_pytorch as cdp

# See change_detection_pytorch/losses
# class JaccardLoss(base.Loss):
#
#     def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels
#
#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return 1 - F.jaccard(
#             y_pr, y_gt,
#             eps=self.eps,
#             threshold=None,
#             ignore_channels=self.ignore_channels,
#         )
#
#
# class DiceLoss(base.Loss):
#
#     def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.beta = beta
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels
#
#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return 1 - F.f_score(
#             y_pr, y_gt,
#             beta=self.beta,
#             eps=self.eps,
#             threshold=None,
#             ignore_channels=self.ignore_channels,
#         )


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass


import torch
class MultiHeadCELoss(torch.nn.Module):
    __name__ = "MultiHeadCELoss"

    def __init__(self,
                 index_weight=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                 weight=None,
                 reduction='mean',
                 loss2=False,
                 loss2_weight=1.0,
                 **kwargs
                 ):
        super(MultiHeadCELoss, self).__init__()
        self.index_weight = index_weight
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.loss_functions = [CrossEntropyLoss(weight=weight, reduction=reduction, **kwargs)
                               for _ in index_weight]
        if self.loss2:
            self.loss_functions_dice = [cdp.losses.DiceLoss(mode='multiclass') for _ in index_weight]

    def forward(self, preds, target):
        losses = []

        for i in range(len(preds)):
            scale_ = preds[i].shape[-1]

            if target.size()[2] == scale_:
                scale_target = target
            else:
                tmp = torch.unsqueeze(target, dim=1)
                tmp = torch.nn.functional.interpolate(tmp.float(), size=[scale_, scale_])
                scale_target = torch.squeeze(tmp, dim=1)
            loss = self.loss_functions[i](preds[i], scale_target.long())

            if self.loss2:
                loss_dice = self.loss_functions_dice[i](preds[i], scale_target.long())
                losses.append(self.index_weight[i] * (loss + loss_dice * self.loss2_weight))
            else:
                losses.append(self.index_weight[i] * loss)

        return sum(losses)
        # return losses[0] + losses[1] + losses[2] + losses[3] + losses[4]



class Multi_MultiHeadCELoss(torch.nn.Module):
    __name__ = "Multi_MultiHeadCELoss"

    def __init__(self,
                 index_weight=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                 weight=None,
                 reduction='mean',
                 loss2=False,
                 loss2_weight=1.0,
                 **kwargs
                 ):
        super(Multi_MultiHeadCELoss, self).__init__()
        self.index_weight = index_weight
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.loss_functions = [CrossEntropyLoss(weight=weight, reduction=reduction, **kwargs)
                               for _ in index_weight]
        if self.loss2:
            self.loss_functions_dice = [cdp.losses.DiceLoss(mode='multiclass') for _ in index_weight]

    def forward(self, preds, target):
        losses = []

        for i in range(len(preds)):
            for j in range(2):
                scale_ = preds[i][j].shape[-1]

                if target.size()[2] == scale_:
                    scale_target = target
                else:
                    tmp = torch.unsqueeze(target, dim=1)
                    tmp = torch.nn.functional.interpolate(tmp.float(), size=[scale_, scale_])
                    scale_target = torch.squeeze(tmp, dim=1)
                loss = self.loss_functions[i](preds[i][j], scale_target.long())

                if self.loss2:
                    loss_dice = self.loss_functions_dice[i](preds[i][j], scale_target.long())
                    losses.append(self.index_weight[i] * (loss + loss_dice * self.loss2_weight))
                else:
                    losses.append(self.index_weight[i] * loss)

        return sum(losses)