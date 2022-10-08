from mmdet3d.models import builder
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors import Base3DDetector
import numpy as np
import torch


@DETECTORS.register_module()
class ANCSH(Base3DDetector):
    JOINT_TYPE_DICT = {0: 'rigid', 1: 'prismatic', 2: 'revolute'}
    def __init__(self,
                 backbone,
                 nocs_head=None,
                 joint_type_head=None,
                 n_max_parts=7,
                 init_cfg=None,
                 **kwargs):
        super(ANCSH, self).__init__(init_cfg)
        self.n_max_parts = n_max_parts

        self.backbone = builder.build_backbone(backbone)

        if nocs_head is not None:
            self.nocs_head = builder.build_head(nocs_head)
        if joint_type_head is not None:
            self.joint_type_backbone = builder.build_backbone(backbone)
            self.joint_type_head = builder.build_head(joint_type_head)
            if 'Vote' in self.joint_type_head._get_name():
                self.dense_joint_type = True
            else:
                self.dense_joint_type = False
        self.part_min_points_num = 128


    def forward(self, return_loss=True, **input):
        if return_loss:
            return self.forward_train(**input)
        else:
            return self.forward_test(**input)

    def forward_train(self, **input):
        P = input['parts_pts']
        if 'parts_pts_feature' in input.keys():
            P_feature = input['parts_pts_feature']
        else:
            P_feature = None
        if P.dim() == 4:
            assert P.shape[1] == 2
            P1 = P[:, 0, :, :]
            P2 = P[:, 1, :, :]

            feat1, feat1_encode = self.backbone(P1, None)
            feat2, feat2_encode = self.backbone(P2, None)

            feat = torch.cat([feat1, feat2], dim=2)
            # feat_encode = torch.stack([feat1_encode, feat2_encode], dim=2)
            feat_encode = [feat1_encode, feat2_encode]
        else:
            feat, feat_encode = self.backbone(P, P_feature)

        pred_dict = self.nocs_head(feat, feat_encode)
        loss_result = self.nocs_head.loss(pred_dict, mode='train', **input)
        if self.with_joint_type:
            points_per_part = input['points_per_part']
            if 'points_per_part_feature' in input.keys():
                points_per_part_feature = input['points_per_part_feature']
            type_pred_dict = dict(joint_type_pred=[])
            for b in range(points_per_part.shape[0]):
                moving_part_ids = input['img_meta'][b]['moving_part_ids']
                # joint_axis = [None] * self.n_max_parts
                joint_type_pred = torch.zeros(self.n_max_parts, 3).to(P.device)
                P_parts = points_per_part[b][moving_part_ids]
                if 'points_per_part_feature' in input.keys():
                    P_parts_feature = points_per_part_feature[b][moving_part_ids]
                else:
                    P_parts_feature = None
                if self.dense_joint_type:
                    part_feat, part_feat_encode = self.joint_type_backbone(P_parts, P_parts_feature, return_decode=True)
                    part_pred = self.joint_type_head(part_feat, None)
                else:
                    _, part_feat_encode = self.joint_type_backbone(P_parts, P_parts_feature, return_decode=False)
                    part_pred = self.joint_type_head(None, part_feat_encode)
                    joint_type_pred[moving_part_ids] = part_pred['joint_type_pred']
                    # type_pred_dict['joint_axis'].append(joint_axis)
                type_pred_dict['joint_type_pred'].append(joint_type_pred)
            pred_dict.update(type_pred_dict)
            loss_result.update(self.joint_type_head.loss_ancsh(pred_dict, mode='train', **input))

        return loss_result

    def forward_test(self, **input):
        if 'parts_pts' not in input:
            return None
        else:
            P = input['parts_pts']
            # P = input['nocs_g'].double()
            if 'parts_pts_feature' in input.keys():
                P_feature = input['parts_pts_feature']
            else:
                P_feature = None
            if P.dim() == 4:
                assert P.shape[1] == 2
                P1 = P[:, 0, :, :]
                P2 = P[:, 1, :, :]

                feat1, feat1_encode = self.backbone(P1, None)
                feat2, feat2_encode = self.backbone(P2, None)

                feat = torch.cat([feat1, feat2], dim=2)
                # feat_encode = torch.cat((feat1_encode, feat2_encode), dim=2)
                feat_encode = [feat1_encode, feat2_encode]
            else:
                feat, feat_encode = self.backbone(P, P_feature)
            # feat = self.backbone(P, P_feature)
            pred_dict = self.nocs_head(feat, feat_encode)
            if self.with_joint_type:
                joint_types_pred = []
                instance_per_point = torch.argmax(pred_dict['W'], dim=2)
                for b in range(instance_per_point.shape[0]):
                    joint_types = [None] * self.n_max_parts
                    part_per_point = instance_per_point[b]
                    part_ids_pred = torch.unique(part_per_point).tolist()
                    for p in part_ids_pred:
                        if p == 0:
                            continue
                        P_part = P[b][part_per_point==p]
                        if 'parts_pts_feature' in input.keys():
                            P_feature = input['parts_pts_feature']
                            P_part_feature = P_feature[b][part_per_point==p]
                        else:
                            P_part_feature = None
                        if P_part.shape[0] < self.part_min_points_num:
                            tile_n = int(self.part_min_points_num / P_part.shape[0]) + 1
                            P_part = torch.cat([P_part] * tile_n, dim=0)
                            if P_part_feature is not None:
                                P_part_feature = torch.cat([P_part_feature] * tile_n, dim=0)
                        perm = np.random.permutation(P_part.shape[0])
                        P_part = P_part[perm[:self.part_min_points_num]]
                        if P_part_feature is not None:
                            P_part_feature = P_part_feature[perm[:self.part_min_points_num]]
                            _, part_feat_encode = self.joint_type_backbone(P_part.unsqueeze(0), P_part_feature.unsqueeze(0), return_decode=False)
                        else:
                            _, part_feat_encode = self.joint_type_backbone(P_part.unsqueeze(0), None, return_decode=False)
                        part_pred = self.joint_type_head(None, part_feat_encode)
                        joint_type_id = torch.argmax(part_pred['joint_type_pred'][0])
                        joint_types[p] = self.JOINT_TYPE_DICT[joint_type_id.item()]
                    joint_types_pred.append(joint_types)
                pred_dict.update(dict(joint_types_pred=joint_types_pred))
            return pred_dict

    @property
    def with_nocs(self):
        return hasattr(self, 'nocs_head') and self.nocs_head is not None

    @property
    def with_joint_type(self):
        return hasattr(self, 'joint_type_head') and self.joint_type_head is not None

