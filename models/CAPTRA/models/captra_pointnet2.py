from mmdet3d.ops import PAConv, PointFPModule, PointSAModuleMSG
from mmcv.ops import three_interpolate, three_nn
from mmcv.runner import BaseModule, force_fp32
from mmcv.ops import GroupAll
import torch.nn as nn
import torch.nn.functional as F
import torch

class PointSAModuleMSG(PointSAModuleMSG):
    def forward(
        self,
        points_xyz,
        features=None,
        indices=None,
        target_xyz=None,
    ):
        new_features_list = []

        # sample points, (B, num_point, 3), (B, num_point)
        new_xyz, indices = self._sample_points(points_xyz, features, indices,
                                               target_xyz)
        for i in range(len(self.groupers)):
            # grouped_results may contain:
            # - grouped_features: (B, C, num_point, nsample)
            # - grouped_xyz: (B, 3, num_point, nsample)
            # - grouped_idx: (B, num_point, nsample)
            grouped_results = self.groupers[i](points_xyz, new_xyz, features)
            # tycoer
            if features is not None:
                if not isinstance(self.groupers[0], GroupAll):
                    grouped_results = torch.cat((grouped_results[:, 3:], grouped_results[:, :3]), dim=1)
            # (B, mlp[-1], num_point, nsample)
            new_features = self.mlps[i](grouped_results)

            # this is a bit hack because PAConv outputs two values
            # we take the first one as feature
            if isinstance(self.mlps[i][0], PAConv):
                assert isinstance(new_features, tuple)
                new_features = new_features[0]

            # (B, mlp[-1], num_point)
            new_features = self._pool_features(new_features)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), indices

class PointFPModule(PointFPModule):
    @force_fp32()
    def forward(self, target: torch.Tensor, source: torch.Tensor,
                target_feats: torch.Tensor,
                source_feats: torch.Tensor) -> torch.Tensor:

        if source is not None:
            dist, idx = three_nn(target, source)
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm

            interpolated_feats = three_interpolate(source_feats, idx, weight)
        else:
            interpolated_feats = source_feats.expand(*source_feats.size()[0:2],
                                                     target.size(1))

        if target_feats is not None:
            # tycoer
            new_features = torch.cat([target_feats, interpolated_feats],
                                     dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlps(new_features)

        return new_features.squeeze(-1)


class CaptraRotNetPointNet2(BaseModule):
    def __init__(self, init_cfg=None):
        BaseModule.__init__(self, init_cfg=init_cfg)
        self.sa1 = PointSAModuleMSG(
            mlp_channels=[[0, 32, 32, 64], [0, 64, 64, 128], [0, 64, 96, 128]],
            num_point=512,
            radii=[0.05, 0.1, 0.2],
            sample_nums=[32, 64, 128],
            use_xyz=True,
            bias=True
        )
        self.sa2 = PointSAModuleMSG(
            mlp_channels=[[320, 128, 128, 256], [320, 128, 196, 256]],
            num_point=128,
            radii=[0.2, 0.4],
            sample_nums=[64, 128],
            use_xyz=True,
            bias=True
        )
        self.sa3 = PointSAModuleMSG(
            mlp_channels=[[512, 256, 512, 1024]],
            num_point=None,
            radii=[None],
            sample_nums=[None],
            use_xyz=True,
            bias=True)

        self.fp3 = PointFPModule(mlp_channels=[1536, 256, 256])
        self.fp2 = PointFPModule(mlp_channels=[576, 256, 128])
        self.fp1 = PointFPModule(mlp_channels=[131, 128, 128])


        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)

    def forward(self, points):
        points = points.transpose(1, 2).contiguous()
        l0_xyz, l0_points = self._split_point_feats(points)
        l1_xyz, l1_points, _ = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points, _ = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_xyz.transpose(1, 2).contiguous(), l1_points)
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        return feat

    @staticmethod
    def _split_point_feats(points):
        xyz = points[..., 0:3].contiguous()
        if points.size(-1) > 3:
            features = points[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None
        return xyz, features


class CaptraCoordNetPointNet2(CaptraRotNetPointNet2):
    def __init__(self, init_cfg=None):
        BaseModule.__init__(self, init_cfg=init_cfg)
        self.sa1 = PointSAModuleMSG(
            mlp_channels=[[3, 32, 32, 64], [3, 64, 64, 128], [3, 64, 96, 128]],
            num_point=512,
            radii=[0.05, 0.1, 0.2],
            sample_nums=[32, 64, 128],
            use_xyz=True,
            bias=True
        )
        self.sa2 = PointSAModuleMSG(
            mlp_channels=[[320, 128, 128, 256], [320, 128, 196, 256]],
            num_point=128,
            radii=[0.2, 0.4],
            sample_nums=[64, 128],
            use_xyz=True,
            bias=True
        )
        self.sa3 = PointSAModuleMSG(
            mlp_channels=[[512, 256, 512, 1024]],
            num_point=None,
            radii=[None],
            sample_nums=[None],
            use_xyz=True,
            bias=True)

        self.fp3 = PointFPModule(mlp_channels=[1536, 256, 256])
        self.fp2 = PointFPModule(mlp_channels=[576, 256, 128])
        self.fp1 = PointFPModule(mlp_channels=[134, 128, 128])


        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)

    def forward(self, points):
        points = points.transpose(1, 2).contiguous()
        l0_xyz, l0_points = self._split_point_feats(points)
        l1_xyz, l1_points, _ = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points, _ = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat((l0_xyz.transpose(1, 2).contiguous(), l0_points), dim=1), l1_points)
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        return feat


    @staticmethod
    def _split_point_feats(points):
        # notice!! In offical model, the pointnet2 for 'CoordNet': features = xyz
        xyz = points[..., 0:3].contiguous()
        features = xyz.transpose(1, 2).contiguous()
        return xyz, features