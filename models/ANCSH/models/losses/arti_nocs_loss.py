import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import LOSSES

DIVISION_EPS = 1e-10


@LOSSES.register_module
class Arti_NOCS_Loss(nn.Module):
    def __init__(self, TYPE_L='L2',
                 MULTI_HEAD=False, SELF_SU=False, contain_bg=True):
        super(Arti_NOCS_Loss, self).__init__()
        self.type = TYPE_L
        self.multi_head = MULTI_HEAD
        self.self_su = SELF_SU
        self.contain_bg = contain_bg

    def forward(self, nocs, nocs_gt, confidence, mask_array=None):
        if self.contain_bg:
            start_part = 1
        else:
            start_part = 0
        n_parts = int(nocs.shape[2] / 3)
        if self.multi_head:
            loss_nocs = 0
            nocs_splits = torch.split(nocs, split_size_or_sections=int(nocs.shape[2] / n_parts), dim=2)
            mask_splits = torch.split(mask_array, split_size_or_sections=int(mask_array.shape[2] / n_parts), dim=2)
            for i in range(start_part, n_parts):
                diff_l2 = torch.norm(nocs_splits[i] - nocs_gt, dim=2)
                diff_abs = torch.sum(torch.abs(nocs_splits[i] - nocs_gt), dim=2)
                if not self.self_su:
                    if self.type == 'L2':
                        loss_nocs += torch.mean(mask_splits[i][:, :, 0] * diff_l2, dim=1)
                    else:
                        loss_nocs += torch.mean(mask_splits[i][:, :, 0] * diff_abs, dim=1)
                else:
                    if self.type == 'L2':
                        loss_nocs += torch.mean(mask_splits[i][:, :, 0] * diff_l2 * confidence[:, :, 0], dim=1)
                    else:
                        loss_nocs += torch.mean(mask_splits[i][:, :, 0] * diff_abs * confidence[:, :, 0], dim=1)
                if self.self_su:
                    loss_nocs += - 0.1 * torch.mean(confidence[:, :, 0].log(), axis=1)

            return loss_nocs

        else:
            diff_l2 = torch.norm(nocs - nocs_gt, dim=2)  # BxN
            diff_abs = torch.sum(torch.abs(nocs - nocs_gt), dim=2)  # BxN
            if not self.self_su:
                if self.type == 'L2':
                    return torch.mean(diff_l2, dim=1)  # B
                else:
                    return torch.mean(diff_abs, dim=1)  # B
            else:
                if self.type == 'L2':
                    return torch.mean(diff_l2 * confidence[:, :, 0] - 0.1 * confidence[:, :, 0].log(), dim=1)  # B
                else:
                    return torch.mean(confidence[:, :, 0] * diff_abs - 0.1 * confidence[:, :, 0].log(), dim=1)  # B