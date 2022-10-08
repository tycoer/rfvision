import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet3d.models.builder import build_backbone, build_head
from mmcv.runner import BaseModule
from mmdet3d.models.builder import DETECTORS, HEADS
from mmdet3d.models.detectors.base import Base3DDetector
from models.Garmentnets.utils.gridding import VirtualGrid

from mmdet3d.models.backbones import PointNet2SASSG
@HEADS.register_module()
class GarmentnetNOCSHead(BaseModule):
    def __init__(self,
                 nocs_bins=64,
                 dropout=True,
                 feature_dim=128,
                 nocs_loss_weight=1,
                 grip_point_loss_weight=1,
                 init_cfg=None,
                 ):
        super().__init__(init_cfg=init_cfg)
        self.nocs_bins = nocs_bins
        self.dropout = dropout
        self.feature_dim = feature_dim
        self.output_dim = 3
        if self.nocs_bins is not None:
            self.output_dim = self.nocs_bins * 3

        self.lin1 = nn.Linear(256, 256)
        self.lin2 = nn.Linear(256, self.feature_dim)
        self.lin3 = nn.Linear(self.feature_dim, self.output_dim)
        self.dp1 = nn.Dropout(p=0.5, inplace=False) if dropout else lambda x: x
        self.dp2 = nn.Dropout(p=0.5, inplace=False) if dropout else lambda x: x

        self.global_conv1 = nn.Conv1d(256, 192, 1, 1)
        self.global_lin1 = nn.Linear(1024, 512)
        self.global_lin2 = nn.Linear(512, 256)
        self.global_lin3 = nn.Linear(256, 1)

        self.global_dp1 = nn.Dropout(p=0.5, inplace=False) if self.dropout else lambda x: x
        self.global_dp2 = nn.Dropout(p=0.5, inplace=False) if self.dropout else lambda x: x
        self.global_dp3 = nn.Dropout(p=0.5, inplace=False) if self.dropout else lambda x: x

        self.vg = self.get_virtual_grid()
        self.criterion = nn.CrossEntropyLoss()
        self.nocs_loss_weight = nocs_loss_weight
        self.grip_point_loss_weight = grip_point_loss_weight

    def forward(self, features, global_features):
        x = F.relu(self.lin1(features))
        x = self.dp1(x)
        x = self.lin2(x)
        features = self.dp2(x)
        logits = self.lin3(features)

        # global prediction
        x = F.relu(self.global_conv1(global_features))
        x = self.global_dp1(x)
        x = self.global_lin1(x)
        x = self.global_dp2(x)
        x = self.global_lin2(x)
        x = self.global_dp3(x)
        global_logits = self.global_lin3(x)
        global_logits = global_logits.squeeze(-1)
        return logits, global_logits, features

    def simple_test(self,
                    pred_logits,
                    pred_global_logits):
        bs, num_points = pred_logits.shape[:2]
        pred_logits_bins = pred_logits.view(bs, num_points, self.nocs_bins, -1).contiguous()
        pred_global_bins = pred_global_logits.view(bs, self.nocs_bins, 3).contiguous()
        # compute confidence
        nocs_bin_idx_pred = torch.argmax(pred_logits_bins, dim=2)
        pred_confidence_bins = F.softmax(pred_logits_bins, dim=2)
        pred_confidence = torch.squeeze(torch.gather(pred_confidence_bins, dim=2, index=torch.unsqueeze(nocs_bin_idx_pred, dim=1)))

        pred_nocs = self.vg.idxs_to_points(nocs_bin_idx_pred)
        grip_bin_idx_pred = torch.argmax(pred_global_bins, dim=1)
        pred_grip_point = self.vg.idxs_to_points(grip_bin_idx_pred)

        results = dict(pred_nocs=pred_nocs,
                       pred_grip_point=pred_grip_point,
                       pred_confidence=pred_confidence,

                       pred_logits_bins=pred_logits_bins,
                       pred_global_bins=pred_global_bins,
                       )

        return results

    def loss(self, pred_logits_bins, pred_global_bins, gt_nocs, gt_grip_point):
        bs, num_points = gt_nocs.shape[:2]
        self.vg.device = gt_nocs.device  # 这里相当于 把nocs 变成voxel
        gt_nocs_idx = self.vg.get_points_grid_idxs(gt_nocs)  # 这里相当于 把nocs 变成voxel
        gt_grip_point_idx = self.vg.get_points_grid_idxs(gt_grip_point)
        loss_nocs = self.nocs_loss_weight * self.criterion(pred_logits_bins.view(bs * num_points, self.nocs_bins, 3),
                                                           gt_nocs_idx.view(bs * num_points, 3))
        loss_grip_point = self.grip_point_loss_weight * self.criterion(pred_global_bins, gt_grip_point_idx.squeeze())
        losses = dict(loss_nocs=loss_nocs,
                      loss_grip_point=loss_grip_point)
        return losses

    def get_virtual_grid(self):
        nocs_bins = self.nocs_bins
        vg = VirtualGrid(lower_corner=(0,0,0), upper_corner=(1,1,1),
            grid_shape=(nocs_bins,)*3, batch_size=1,
            # device=self.device,
            int_dtype=torch.int64,
            float_dtype=torch.float32)
        return vg

@DETECTORS.register_module()
class GarmentnetNOCS(Base3DDetector):
    def __init__(self,
                 backbone,
                 head,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

    def forward_train(self,
                      points,
                      gt_nocs,
                      gt_grip_point,
                      **kwargs):
        feats = self.backbone(points)

        pred_logits, pred_global_logits, _ = self.head(
            feats['fp_features'][-1].transpose(1, 2),
            feats['sa_features'][-1])
        results = self.head.simple_test(pred_logits=pred_logits, pred_global_logits=pred_global_logits)
        losses = self.head.loss(results['pred_logits_bins'],
                                results['pred_global_bins'],
                                gt_nocs,
                                gt_grip_point)
        nocs_err_distance = torch.norm(results['pred_nocs'] - gt_nocs, dim=-1).mean().detach()
        grip_point_err_distance = torch.norm(results['pred_grip_point'] - gt_grip_point, dim=-1).mean().detach()
        losses['nocs_err_distance'] = nocs_err_distance
        losses['grip_point_err_distance'] = grip_point_err_distance

        return losses

    def simple_test(self, points):
        feats = self.backbone(points)
        pred_logits, pred_global_logits, features = self.head(
            feats['fp_features'][-1].transpose(1, 2).contiguous().view(-1, feats['fp_features'][-1].shape[1]),
            feats['sa_features'][-1]
        )
        results = self.head.simple_test(pred_logits=pred_logits, pred_global_logits=pred_global_logits)
        results['per_point_features'] = features
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def extract_feat(self, imgs):
        pass