from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

__all__ = ["SoftWeightCEWithLogitsLoss"]


class SoftWeightCEWithLogitsLoss(nn.Module):
    __name__ = "SoftWeightCEWithLogitsLoss"

    __constants__ = ["weight", "pos_weight", "reduction", "ignore_index", "smooth_factor"]

    def __init__(
            self,
            weight: Optional[torch.Tensor] = None,
            ignore_index: Optional[int] = -100,
            reduction: str = "mean",
            smooth_factor: Optional[float] = None,
            pos_weight: Optional[torch.Tensor] = None,
    ):
        """Drop-in replacement for torch.nn.BCEWithLogitsLoss with few additions: ignore_index and label_smoothing

        Args:
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])

        Shape
             - **y_pred** - torch.Tensor of shape NxCxHxW
             - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)

        Returns:
            loss: torch.Tensor
        """

        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)
        else:
            soft_targets = y_true

        loss = F.cross_entropy(y_pred, soft_targets, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

        return loss
