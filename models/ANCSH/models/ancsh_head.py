import os, sys

import torch
import torch.nn as nn

from mmdet3d.models.builder import HEADS, build_loss
import torch
import random
import numpy as np


def sample_parts(gocs, n_parts, size=2):
    sampled_part_ids = random.sample(range(1, n_parts), size)

    return sampled_part_ids


def sample_joint(joint_cls, n_joints, size=1):
    sampled_joint_id = random.sample(range(1, n_joints), size)

    return sampled_joint_id


@HEADS.register_module()
class ANCSHHead(nn.Module):
    def __init__(self,
                 n_max_parts,
                 mixed_pred=False,
                 part_relation=False,
                 loss_miou=None,
                 loss_nocs=None,
                 loss_vect=None,
                 loss_weights=(10.0, 1.0, 1.0, 5.0, 5.0, 0.2, 1.0, 1.0)):
        super(ANCSHHead, self).__init__()
        self.n_max_parts = n_max_parts
        # self.n_joints = int((n_max_parts - 1) * (n_max_parts - 2) / 2 + 1)
        self.mixed_pred = mixed_pred
        self.part_relation = part_relation

        out_dims = [self.n_max_parts, 3 * self.n_max_parts]
        if mixed_pred:
            out_dims.append(1 * self.n_max_parts) # scale
            out_dims.append(3 * self.n_max_parts) # translation
        out_dims.append(1)

        in_dim = 128
        self.conv_W = nn.Sequential(
            nn.Conv1d(in_dim, self.n_max_parts, 1, padding=0),
            # nn.BatchNorm1d(n_max_parts)

        )
        if self.mixed_pred:
            self.conv_scale = nn.Sequential(
                nn.Conv1d(in_dim, self.n_max_parts, 1, padding=0),
                # nn.BatchNorm1d(n_max_parts)
            )
            self.conv_trans = nn.Sequential(
                nn.Conv1d(in_dim, 3 * self.n_max_parts, 1, padding=0),
                # nn.BatchNorm1d(3 * n_max_parts)
            )
        self.conv_confi = nn.Sequential(
            nn.Conv1d(in_dim, 1, 1, padding=0),
            # nn.BatchNorm1d(1)
        )

        if self.mixed_pred:
            print('Now we are using early_split_nocs')
            self.conv_nocs = nn.Sequential(
                nn.Conv1d(in_dim, 128, 1, padding=0),
                # nn.BatchNorm1d(128),
                nn.Conv1d(in_dim, 3 * self.n_max_parts, 1, padding=0),
                # nn.BatchNorm1d(3 * n_max_parts),
            )

        else:
            self.conv_nocs = nn.Sequential(
                nn.Conv1d(in_dim, 3 * self.n_max_parts, 1, padding=0),
                # nn.BatchNorm1d(3 * n_max_parts),
            )

        joint_layer_dims = [128, 128]
        self.conv_joint_feat = nn.Sequential(
            nn.Conv1d(in_dim, joint_layer_dims[0], 1, padding=0),
            nn.BatchNorm1d(joint_layer_dims[0]),
            nn.Dropout(p=0.5),

            nn.Conv1d(in_dim, joint_layer_dims[1], 1, padding=0),
            nn.BatchNorm1d(joint_layer_dims[1]),
            nn.Dropout(p=0.5)
        )

        self.conv_joint_axis = nn.Conv1d(joint_layer_dims[1], 3, 1, padding=0)
        self.conv_univect = nn.Conv1d(joint_layer_dims[1], 3, 1, padding=0)
        self.conv_heatmap = nn.Conv1d(joint_layer_dims[1], 1, 1, padding=0)
        self.conv_joint_cls = nn.Conv1d(joint_layer_dims[1], self.n_max_parts, 1, padding=0)

        self.loss_miou = build_loss(loss_miou)
        self.loss_nocs = build_loss(loss_nocs)
        self.loss_vect = build_loss(loss_vect)

        self.nocs_loss_multiplier, self.miou_loss_multiplier, self.gocs_loss_multiplier, \
        self.heatmap_loss_multiplier, self.unitvec_loss_multiplier, self.orient_loss_multiplier, \
        self.index_loss_multiplier, self.total_loss_multiplier = loss_weights

    def forward(self, x, x_encode):
        W = self.conv_W(x).permute(0,2,1)
        nocs_per_points = self.conv_nocs(x).permute(0,2,1)

        confi_per_points = self.conv_confi(x).permute(0,2,1)

        if self.mixed_pred:
            scale_per_points = self.conv_scale(x).permute(0,2,1)
            trans_per_points = self.conv_trans(x).permute(0,2,1)
            scale_per_points = torch.sigmoid(scale_per_points)
            trans_per_points = torch.tanh(trans_per_points)

        if self.loss_miou._get_name() == 'Arti_MIoU_Loss':
            W = torch.softmax(W, dim=2)  # BxNxK # maximum
        confi_per_points = torch.sigmoid(confi_per_points)
        nocs_per_points = torch.sigmoid(nocs_per_points)  # BxNx3

        x_joint = self.conv_joint_feat(x)
        heatmap = self.conv_heatmap(x_joint).permute(0,2,1)
        unitvec = self.conv_univect(x_joint).permute(0,2,1)
        joint_axis = self.conv_joint_axis(x_joint).permute(0,2,1)
        joint_cls = self.conv_joint_cls(x_joint).permute(0,2,1)

        heatmap = torch.sigmoid(heatmap)
        unitvec = torch.tanh(unitvec)
        joint_axis = torch.tanh(joint_axis)
        if self.loss_miou._get_name() == 'Arti_MIoU_Loss':
            joint_cls = torch.softmax(joint_cls, dim=2)

        pred = {
            'W': W,
            'nocs_per_point': nocs_per_points,
            'confi_per_point': confi_per_points,
            'heatmap_per_point': heatmap,
            'unitvec_per_point': unitvec,
            'joint_axis_per_point': joint_axis,
            'index_per_point': joint_cls
        }

        if self.mixed_pred:
            scale_per_points_tiled = scale_per_points.unsqueeze(-1).repeat(1, 1, 1, 3).view([
                scale_per_points.shape[0], scale_per_points.shape[1], 3 * self.n_max_parts
            ])

            trans_per_points_tiled = trans_per_points


            # scale_per_points_tiled = tf.tile(scale_per_points, [1, 1, 3*n_max_parts])
            # trans_per_points_tiled = tf.tile(trans_per_points, [1, 1, n_max_parts])

            #assert trans_per_points_tiled.get_shape().as_list()[2] == scale_per_points_tiled.get_shape().as_list()[
                #2] == 3 * n_max_parts, print(scale_per_points_tiled.get_shape().as_list()[2],
                                             #trans_per_points_tiled.get_shape().as_list()[2])
            gocs_per_points = nocs_per_points * scale_per_points_tiled + trans_per_points_tiled
            pred['gocs_per_point'] = gocs_per_points
            pred['global_scale'] = scale_per_points
            pred['global_translation'] = trans_per_points

        if self.part_relation:
            sampled_parts = sample_parts(gocs_per_points, self.n_max_parts)
            sampled_joint = sample_joint(joint_cls, self.n_joints)

        return pred

    def loss(self, pred_dict, mode='train', **input):
        W = pred_dict['W']
        batch_size = W.shape[0]  # B*N*K(k parts)
        n_points = W.shape[1]
        assert W.shape[2] == self.n_max_parts  # n_max_parts should not be dynamic, fixed number of parts

        # n_registered_primitives = fitter_factory.get_n_registered_primitives()

        # note that I_gt can contain -1, indicating part of unknown primitive type
        I_gt = input['parts_cls'].squeeze(-1)  # BxN
        # n_parts_gt = torch.max(I_gt, 1)[0] + 1
        # mask_gt =

        if self.loss_miou._get_name() == 'Arti_MIoU_Loss':
            miou_loss = self.loss_miou(W, I_gt)
        else: # use cross entropy
            miou_loss = self.loss_miou(W.permute(0, 2, 1), I_gt.long())

        nocs_loss = self.loss_nocs(pred_dict['nocs_per_point'],
                                    input['nocs_p'],
                                    pred_dict['confi_per_point'],
                                    mask_array=input['mask_array'])

        if self.mixed_pred:
            gocs_loss = self.loss_nocs(pred_dict['gocs_per_point'],
                                               input['nocs_g'],
                                               pred_dict['confi_per_point'],
                                               mask_array=input['mask_array'])

        heatmap_loss = self.loss_vect(pred_dict['heatmap_per_point'],
                                              input['offset_heatmap'].squeeze(),
                                              confidence=input['joint_cls_mask'].squeeze(-1))

        unitvec_loss = self.loss_vect(pred_dict['unitvec_per_point'],
                                              input['offset_unitvec'],
                                              confidence=input['joint_cls_mask'].squeeze(-1))

        orient_loss = self.loss_vect(pred_dict['joint_axis_per_point'],
                                             input['joint_orient'],
                                             confidence=input['joint_cls_mask'].squeeze(-1))

        J_gt = input['joint_cls'].squeeze(-1)
        inds_pred = pred_dict['index_per_point']
        if self.loss_miou._get_name() == 'Arti_MIoU_Loss':
            miou_joint_loss = self.loss_miou(inds_pred, J_gt)[:, :-1]
        else: # use cross entropy
            miou_joint_loss = self.loss_miou(inds_pred.permute(0,2,1), J_gt.long())

        loss_dict = {
            'nocs_loss': nocs_loss,
            'miou_loss': miou_loss,
            'heatmap_loss': heatmap_loss,
            'unitvec_loss': unitvec_loss,
            'orient_loss' : orient_loss,
            'index_loss'  : miou_joint_loss
            }

        if self.mixed_pred:
            loss_dict['gocs_loss'] = gocs_loss

        loss = self.collect_losses(loss_dict)
        return loss

        if mode == 'train':
            result = {'loss': loss}
            return result
        elif mode == 'test':
            loss_dict.update({'total_loss': self.collect_losses(loss_dict)})
            return loss_dict
        else:
            ValueError('mode must be train or test')

    def collect_losses_fuse(self, loss_dict):
        total_loss = 0

        nocs_loss_per_part = loss_dict['nocs_loss']
        total_nocs_loss = torch.mean(nocs_loss_per_part)

        if self.mixed_pred:
            gocs_loss_per_part = loss_dict['gocs_loss']
            total_gocs_loss = torch.mean(gocs_loss_per_part)

        total_heatmap_loss = torch.mean(loss_dict['heatmap_loss'])
        total_unitvec_loss = torch.mean(loss_dict['unitvec_loss'])
        total_orient_loss = torch.mean(loss_dict['orient_loss'])

        total_index_loss = torch.mean(loss_dict['index_loss'])

        miou_loss_per_part = loss_dict['miou_loss']
        total_miou_loss = torch.mean(miou_loss_per_part)

        total_loss += self.nocs_loss_multiplier * total_nocs_loss
        total_loss += self.miou_loss_multiplier * total_miou_loss

        if self.mixed_pred:
            total_loss += self.gocs_loss_multiplier * total_gocs_loss

        if self.mixed_pred:  # only use it in part + global NOCS
            total_loss += self.heatmap_loss_multiplier * total_heatmap_loss
            total_loss += self.unitvec_loss_multiplier * total_unitvec_loss
            total_loss += self.orient_loss_multiplier * total_orient_loss
            total_loss += self.index_loss_multiplier * total_index_loss

        total_loss *= self.total_loss_multiplier

        return total_loss

    def collect_losses(self, loss_dict):
        loss_dict['nocs_loss'] *= self.nocs_loss_multiplier
        loss_dict['miou_loss'] *= self.miou_loss_multiplier
        loss_dict['gocs_loss'] *= self.gocs_loss_multiplier

        loss_dict['heatmap_loss'] *= self.heatmap_loss_multiplier
        loss_dict['unitvec_loss'] *= self.unitvec_loss_multiplier
        loss_dict['orient_loss'] *= self.orient_loss_multiplier
        loss_dict['index_loss'] *= self.index_loss_multiplier

        return loss_dict
