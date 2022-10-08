import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import LOSSES

DIVISION_EPS = 1e-10


@LOSSES.register_module()
class Arti_MIoU_Loss(nn.Module):
    def __init__(self, reorder=False, contain_bg=True):
        super(Arti_MIoU_Loss, self).__init__()
        self.reorder = reorder
        self.contain_bg = contain_bg

    def forward(self, W, I_gt, matching_indices=None):
        if self.reorder:
            W_reordered = batched_gather(W, indices=matching_indices, axis=2)
            # W_reordered = W
        else:
            W_reordered = W

        depth = W.shape[2]
        W_gt = F.one_hot(I_gt.long(), num_classes=depth).float()
        dot = torch.sum(W_gt * W_reordered, 1)
        denominator = torch.sum(W_gt, 1) + torch.sum(W_reordered, 1) - dot
        mIoU = dot / (denominator + DIVISION_EPS)
        return 1.0 - mIoU