import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import LOSSES

DIVISION_EPS = 1e-10


@LOSSES.register_module
class Arti_VECT_Loss(nn.Module):
    def __init__(self, TYPE_L='L2',
                 MULTI_HEAD=False, SELF_SU=False, contain_bg=True):
        super(Arti_VECT_Loss, self).__init__()
        self.type = TYPE_L
        self.multi_head = MULTI_HEAD
        self.self_su = SELF_SU
        self.contain_bg = contain_bg

    def forward(self, vect, vect_gt, confidence=None, mask_array=None):
        if self.multi_head:
            if vect_gt.shape[2] == 1:
                n_parts = int(vect.shape[2])
            else:
                n_parts = int(vect.shape[2] / 3)
            start_part = 1
            loss_vect = 0
            vect_splits = torch.split(vect, split_size_or_sections=int(vect.shape[2] / n_parts), dim=2)
            mask_splits = torch.split(mask_array, split_size_or_sections=int(mask_array.shape[2] / n_parts), dim=2)
            for i in range(start_part, n_parts):
                diff_l2 = torch.norm(vect_splits[i] - vect_gt, dim=2)  # BxN
                diff_abs = torch.sum(torch.abs(vect_splits[i] - vect_gt), dim=2)
                if not self.self_su:
                    if self.type == 'L2':
                        loss_vect += torch.mean(mask_splits[i][:, :, 0] * diff_l2, dim=1)
                    else:
                        loss_vect += torch.mean(mask_splits[i][:, :, 0] * diff_abs, dim=1)
                else:
                    if self.type == 'L2':
                        loss_vect += torch.mean(mask_splits[i][:, :, 0] * diff_l2 * confidence[:, :, 0], dim=1)
                    else:
                        loss_vect += torch.mean(mask_splits[i][:, :, 0] * diff_abs * confidence[:, :, 0], dim=1)
                if self.self_su:
                    loss_vect += - 0.01 * torch.mean(confidence[:, :, 0].log(), dim=1)

            return loss_vect

        else:
            if vect.shape[2] == 1:
                vect = torch.squeeze(vect, 2)
                if confidence is not None:
                    diff_l2 = torch.abs(vect - vect_gt) * confidence
                    diff_abs = torch.abs(vect - vect_gt) * confidence
                else:
                    diff_l2 = torch.abs(vect - vect_gt)
                    diff_abs = torch.abs(vect - vect_gt)
            else:
                if confidence is not None:
                    diff_l2 = torch.norm(vect - vect_gt, dim=2) * confidence
                    diff_abs = torch.sum(torch.abs(vect - vect_gt), dim=2) * confidence
                else:
                    diff_l2 = torch.norm(vect - vect_gt, dim=2)
                    diff_abs = torch.sum(torch.abs(vect - vect_gt), dim=2)
            if not self.self_su:
                if self.type == 'L2':
                    return torch.mean(diff_l2, dim=1)
                else:
                    return torch.mean(diff_abs, dim=1)
            else:
                if self.type == 'L2':
                    return torch.mean(diff_l2 * confidence[:, :, 0] - 0.01 * confidence[:, :, 0].log(), dim=1)
                else:
                    return torch.mean(confidence[:, :, 0] * diff_abs - 0.01 * confidence[:, :, 0].log(), dim=1)