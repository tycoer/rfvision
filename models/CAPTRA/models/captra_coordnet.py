import torch.nn as nn
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors import Base3DDetector
import torch
import numpy as np
from .captra_backbone import CoordNet
from models.CAPTRA.utils.part_dof_utils import part_model_batch_to_part, eval_part_full, add_noise_to_part_dof, \
    compute_parts_delta_pose
from models.CAPTRA.utils.utils import cvt_torch
from models.CAPTRA.models.loss import compute_part_dof_loss, compute_point_pose_loss, compute_miou_loss, compute_nocs_loss
from models.CAPTRA.utils.bbox_utils import yaxis_from_corners, tensor_bbox_from_corners


@DETECTORS.register_module()
class CaptraCoordNet(Base3DDetector):
    def __init__(self,
                 obj_cfg,
                 backbone_out_dim=128,
                 extra_dims=1,
                 nocs_head_dims=[128],
                 pwm_num = 128,
                 pose_perturb_cfg=dict(type='normal',
                                       scale=0.02,
                                       translation=0.03,
                                       rotation=5.0),
                 loss_t_weight=5.0,
                 loss_s_weight=5.0,
                 loss_corner_weight=10,
                 loss_nocs_weight=10,
                 loss_seg_weight=1.0,
                 loss_nocs_dist_weight=5.0,
                 loss_nocs_pwm_weight=5.0,

                 init_cfg=None,
                 **kwargs,
                 ):
        super(CaptraCoordNet, self).__init__(init_cfg)
        self.coordnet = CoordNet(obj_cfg, backbone_out_dim, extra_dims, nocs_head_dims)
        self.obj_cfg = obj_cfg
        self.tree = self.obj_cfg['tree']
        self.root = [p for p in range(len(self.tree)) if self.tree[p] == -1][0]
        self.sym =  self.obj_cfg['sym']
        self.pwm_num = None if not self.sym else pwm_num
        self.pose_loss_type = {'r': 'frob', 's': 'l1', 't': 'l1', 'point': 'l1'}
        self.pose_perturb_cfg = pose_perturb_cfg
        self.pose_perturb_cfg['rotation'] = np.deg2rad(pose_perturb_cfg['rotation'])


        self.num_parts = self.obj_cfg['num_parts']
        self.sym = self.obj_cfg['sym']

        self.loss_t_weight = loss_t_weight
        self.loss_s_weight = loss_s_weight
        self.loss_corner_weight = loss_corner_weight
        self.loss_nocs_weight = loss_nocs_weight
        self.loss_seg_weight = loss_seg_weight
        self.loss_nocs_dist_weight = loss_nocs_dist_weight
        self.loss_nocs_pwm_weight = loss_nocs_pwm_weight

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


    def forward_train(self, **data):
        losses = {}
        data = self.recover_data(**data)
        feed_dict = self.prepare_data(self.set_data(data))
        pred_dict = self.coordnet(feed_dict)
        losses_ = self.loss(pred_dict, feed_dict)

        losses['loss_t'] = self.loss_t_weight * losses_['tloss']
        losses['loss_s'] = self.loss_s_weight * losses_['sloss']
        losses['loss_corner'] = self.loss_corner_weight * losses_['corner_loss']
        losses['loss_seg'] = self.loss_seg_weight * losses_['tloss']
        if self.sym:
            losses['loss_nocs_dist'] = self.loss_nocs_dist_weight * losses_['nocs_dist_loss']
            losses['loss_nocs_pwm'] = self.loss_nocs_pwm_weight * losses_['nocs_pwm_loss']
        else:
            losses['loss_nocs'] = self.loss_nocs_weight * losses_['nocs_loss']
        return losses

    def forward(self, return_loss=True, **data):
        self.device = data['points'].device
        if return_loss:
            return self.forward_train(**data)
        else: return self.forward_test(**data)


    def forward_test(self, **data):
        feed_dict = self.prepare_data(**data)
        pred_dict = self.coordnet(feed_dict)
        return pred_dict

    def loss(self, pred_dict, feed_dict, test=False, per_instance=False):
        loss_dict = {}

        seg_loss = compute_miou_loss(pred_dict['seg'], feed_dict['labels'],
                                     per_instance=False)
        loss_dict['seg_loss'] = seg_loss

        gt_labels = feed_dict['labels']
        pred_labels = torch.max(pred_dict['seg'], dim=-2)[1]  # [B, P, N] -> [B, N]
        labels = pred_labels if test else gt_labels

        nocs_loss = compute_nocs_loss(pred_dict['nocs'], feed_dict['nocs'],
                                      labels=labels,
                                      confidence=None, loss='l2', self_supervise=False,
                                      per_instance=False, sym=self.sym, pwm_num=self.pwm_num)
        if self.sym:
            loss_dict['nocs_dist_loss'], loss_dict['nocs_pwm_loss'] = nocs_loss
        else:
            loss_dict['nocs_loss'] = nocs_loss

        gt_corners = feed_dict['meta']['nocs_corners'].float().to(self.device)
        if self.sym:
            gt_bbox = yaxis_from_corners(gt_corners, self.device)
        else:
            gt_bbox = tensor_bbox_from_corners(gt_corners, self.device)

        pose_diff, per_diff = eval_part_full(feed_dict['gt_part'], pred_dict['part'],
                                             yaxis_only=self.sym)
        init_part_pose = feed_dict['init_part']
        init_pose_diff, init_per_diff = eval_part_full(feed_dict['gt_part'], init_part_pose,
                                                       yaxis_only=self.sym)
        loss_dict.update(pose_diff)
        loss_dict.update({f'init_{key}': value for key, value in init_pose_diff.items()})

        """loss"""

        loss_dict.update(compute_part_dof_loss(feed_dict['gt_part'], pred_dict['part'],
                                               self.pose_loss_type))

        corner_loss, corner_per_diff = compute_point_pose_loss(feed_dict['gt_part'], pred_dict['part'],
                                                               gt_bbox,
                                                               metric=self.pose_loss_type['point'])
        loss_dict['corner_loss'] = corner_loss
        return loss_dict

    def prepare_data(self, data):
        feed_dict = data

        gt_part, init_part = self.prepare_poses(data)

        canon_pose = {key: init_part[key][:, self.root] for key in ['rotation', 'translation', 'scale']}

        feed_dict['canon_pose'] = canon_pose
        feed_dict['init_part'] = init_part
        feed_dict['gt_part'] = gt_part
        return feed_dict

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

    def set_data(self, data):
        feed_dict = {}
        for key, item in data.items():
            if key not in ['meta', 'labels', 'points', 'nocs']:
                continue
            elif key in ['meta']:
                pass
            elif key in ['labels']:
                item = item.long().to(self.device)
            else:
                item = item.float().to(self.device)
            feed_dict[key] = item
        feed_dict['points_mean'] = data['meta']['points_mean'].float().to(self.device)
        return feed_dict

    def simple_test(self, img, img_metas, **kwargs):
        pass

    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def extract_feat(self, imgs):
        pass