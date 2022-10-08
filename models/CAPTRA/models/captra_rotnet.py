import torch.nn as nn
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors import Base3DDetector
import torch
import numpy as np
from .captra_backbone import PartCanonNet
from models.CAPTRA.utils.part_dof_utils import part_model_batch_to_part, eval_part_full, add_noise_to_part_dof, \
    compute_parts_delta_pose
from models.CAPTRA.utils.utils import cvt_torch
from models.CAPTRA.models.loss import compute_part_dof_loss, compute_point_pose_loss, rot_yaxis_loss, rot_trace_loss
from models.CAPTRA.utils.bbox_utils import yaxis_from_corners, tensor_bbox_from_corners

@DETECTORS.register_module()
class CaptraRotNet(Base3DDetector):
    def __init__(self,
                 obj_cfg,
                 pose_perturb_cfg=dict(type='normal',
                                       scale=0.02,
                                       translation=0.03,
                                       rotation=5.0),
                 backbone_out_dim=128,
                 network_type='rot',
                 loss_r_weight=10,
                 loss_corner_weight=1,
                 init_cfg=None,
                 **kwargs
                 ):
        super(CaptraRotNet, self).__init__(init_cfg)
        self.loss_r_weight = loss_r_weight
        self.loss_corner_weight = loss_corner_weight
        self.rotnet = PartCanonNet(obj_cfg, network_type, backbone_out_dim)
        self.obj_cfg = obj_cfg
        self.pose_perturb_cfg = pose_perturb_cfg
        self.pose_perturb_cfg['rotation'] = np.deg2rad(pose_perturb_cfg['rotation'])
        self.pose_loss_type = {'r': 'frob', 's': 'l1', 't': 'l1', 'point': 'l1'}


        self.num_parts = self.obj_cfg['num_parts']
        self.sym = self.obj_cfg['sym']


    def forward_train(self, **data):
        losses = {}
        data = self.recover_data(**data)
        feed_dict = self.prepare_data(data)
        pred_dict = self.rotnet(feed_dict)
        losses_ = self.loss(pred_dict, feed_dict)
        losses['loss_r'] = self.loss_r_weight * losses_['rloss']
        losses['loss_corner'] = self.loss_r_weight * losses_['corner_loss']
        return losses

    def forward(self, return_loss=True, **data):
        self.device = data['points'].device
        if return_loss:
            return self.forward_train(**data)
        else: return self.forward_test(**data)


    def forward_test(self, **data):
        feed_dict = self.prepare_data(**data)
        pred_dict = self.rotnet(feed_dict)
        return pred_dict

    def loss(self, pred_dict, feed_dict, test_mode=False, per_instance=False):
        loss_dict = {}

        labels = feed_dict['labels']

        gt_corners = feed_dict['meta']['nocs_corners'].float().to(self.device)
        if self.sym:
            gt_bbox = yaxis_from_corners(gt_corners, self.device)
        else:
            gt_bbox = tensor_bbox_from_corners(gt_corners, self.device)

        pose_diff, per_diff = eval_part_full(feed_dict['gt_part'], pred_dict['part'],
                                             per_instance=per_instance, yaxis_only=self.sym)
        init_part_pose = feed_dict['state']['part']
        init_pose_diff, init_per_diff = eval_part_full(feed_dict['gt_part'], init_part_pose,
                                                       per_instance=per_instance,
                                                       yaxis_only=self.sym)
        loss_dict.update(pose_diff)
        loss_dict.update({f'init_{key}': value for key, value in init_pose_diff.items()})
        if per_instance:
            per_diff.update({f'init_{key}': value for key, value in init_per_diff.items()})
            self.record_per_diff(feed_dict, {'test': per_diff})

        loss_dict.update(compute_part_dof_loss(feed_dict['gt_part'], pred_dict['part'],
                                               self.pose_loss_type))

        corner_loss, corner_per_diff = compute_point_pose_loss(feed_dict['gt_part'], pred_dict['part'],
                                                               gt_bbox,
                                                               metric=self.pose_loss_type['point'])
        loss_dict['corner_loss'] = corner_loss

        root_delta = feed_dict['root_delta']
        eye_mat = torch.cat([torch.eye(self.num_parts), torch.zeros(2, self.num_parts)], dim=0)
        part_mask = eye_mat[labels,].to(labels.device).transpose(-1, -2)  # [B, P, N]

        def masked_mean(value, mask):
            return torch.sum(value * mask) / torch.clamp(torch.sum(mask), min=1.0)

        if 'point_rotation' in pred_dict:
            point_rotation = pred_dict['point_rotation']  # [B, P, N, 3, 3]
            gt_rotation = root_delta['rotation'].unsqueeze(-3)  # [B, P, 1, 3, 3]
            if point_rotation.shape[1] == 1 and len(point_rotation.shape) > len(gt_rotation.shape):
                point_rotation = point_rotation.squeeze(1)  # [B, N, 3, 3]
            if self.sym:
                rloss = rot_yaxis_loss(gt_rotation, point_rotation)
            else:
                rloss = rot_trace_loss(gt_rotation, point_rotation, metric=self.pose_loss_type['r'])
            loss_dict['rloss'] = masked_mean(rloss, part_mask)
        return loss_dict

    def prepare_data(self, data):
        gt_part, init_part = self.prepare_poses(data)

        input = {'points': data['points'] ,
                 'points_mean': data['meta']['points_mean'],
                 'nocs': data['nocs'],
                 'state': {'part': init_part}, 'gt_part': gt_part}

        input = cvt_torch(input, self.device)
        input['meta'] = data['meta']
        input['labels'] = data['labels'].long().to(self.device)

        part_pose = input['state']['part']
        canon_pose = {key: part_pose[key].reshape((-1, ) + part_pose[key].shape[2:])  # [B, P, x] --> [B * P, x]
                      for key in ['rotation', 'translation', 'scale']}

        input['canon_pose'] = canon_pose

        batch_size = len(input['gt_part']['scale'])
        part_delta = compute_parts_delta_pose(input['state']['part'],
                                              input['gt_part'],
                                              {key: value.reshape((batch_size, self.num_parts) + value.shape[1:])
                                               for key, value in canon_pose.items()})
        input['root_delta'] = part_delta
        return input

    def prepare_poses(self, data):
        device = data['points'].device
        gt_part = part_model_batch_to_part(cvt_torch(data['meta']['nocs2camera'], device),
                                           self.obj_cfg['num_parts'],
                                           device)
        init_part = add_noise_to_part_dof(gt_part, self.pose_perturb_cfg)
        if 'crop_pose' in data['meta']:
            crop_pose = part_model_batch_to_part(cvt_torch(data['meta']['crop_pose'], device),
                                                 self.obj_cfg['num_parts'], device)
            for key in ['translation', 'scale']:
                init_part[key] = crop_pose[key]
        return gt_part, init_part

    def recover_data(self, **data):

        if data['nocs2camera_rotation'].ndim == 4:
            nocs2camera = []
            for s, r, t in zip(data['nocs2camera_scale'].transpose(0, 1).contiguous(),
                               data['nocs2camera_rotation'].transpose(0, 1).contiguous(),
                               data['nocs2camera_translation'].transpose(0, 1).contiguous()):
                nocs2camera.append(dict(rotation=r, scale=s, translation=t))
        else:
            nocs2camera = [dict(rotation=data['nocs2camera_rotation'],
                                scale=data['nocs2camera_scale'],
                                translation=data['nocs2camera_translation'])]
        data = dict(points=data['points'],
                    labels=data['labels'],
                    nocs=data['nocs'],
                    meta=dict(nocs2camera=nocs2camera,
                                points_mean=data['points_mean'],
                                nocs_corners=data['nocs_corners'],
                                ))

        if 'crop_pose_scale' in data and 'crop_pose_translation' in data:
            data['meta']['crop_pose'] = [dict(scale=data['crop_pose_scale'],
                                             translation=data['crop_pose_translation'])],
        return data

    def simple_test(self, img, img_metas, **kwargs):
        pass

    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def extract_feat(self, imgs):
        pass

