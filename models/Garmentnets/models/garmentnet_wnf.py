import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet3d.models.builder import build_backbone, build_head, build_detector
from mmdet3d.models.builder import DETECTORS, HEADS
from mmdet3d.models.detectors.base import Base3DDetector
import torch_scatter

from models.Garmentnets.utils.gridding import VirtualGrid

@DETECTORS.register_module()
class GarmentnetWNF(Base3DDetector):
    def __init__(self,
                 backbone,
                 volume_agg_params,
                 volume_decoder_params,
                 surface_decoder_params,
                 garmentnet_nocs=None,
                 volume_loss_weight=1.0,
                 surface_loss_weight=1.0,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        # model init
        self.garmentnet_nocs = build_detector(garmentnet_nocs)
        self.garmentnet_nocs.eval()
        self.garmentnet_nocs.requires_grad_(False)

        self.unet_3d = build_backbone(backbone)

        self.volume_decoder = build_head(volume_decoder_params)
        self.surface_decoder = build_head(surface_decoder_params)
        self.volume_agg = build_head(volume_agg_params)
        # loss init
        self.criterion = nn.MSELoss(reduction='mean')
        self.volume_loss_weight = volume_loss_weight
        self.surface_loss_weight = surface_loss_weight

    def forward_train(self,
                      points,
                      volume_query_points,
                      surface_query_points,
                      gt_volume_value,
                      gt_sim_points,
                      **kwargs):
        nocs_results = self.garmentnet_nocs.simple_test(points)

        batch_size, num_points, feature_num = points.shape
        batch_idx = (torch.ones((batch_size, num_points)) * torch.arange(batch_size)[:, None]).flatten().to(dtype=torch.int64, device=points.device)
        # flatten inputs
        xyz = points[..., :3].view(-1, 3).contiguous()
        volume_query_points = volume_query_points.squeeze(1)
        surface_query_points = surface_query_points.squeeze(1)
        gt_volume_value = gt_volume_value.squeeze(1)
        gt_sim_points = gt_sim_points.squeeze(1)
        unet3d_result = self.unet3d_forward(
            xyz=xyz,
            pred_nocs=nocs_results['pred_nocs'],
            per_point_features=nocs_results['per_point_features'],
            pred_confidence=nocs_results['pred_confidence'],
            batch_size=batch_size,
            batch_idx=batch_idx
        )
        volume_decoder_result = self.volume_decoder_forward(
            unet3d_result, volume_query_points)  # (volume_query_points 2, 6000, 3)
        surface_decoder_result = self.surface_decoder_forward(
            unet3d_result, surface_query_points)

        losses = self.loss(pred_sim_points=surface_decoder_result['out_features'],
                           pred_volume_value=volume_decoder_result['pred_volume_value'],
                           gt_sim_points=gt_sim_points,
                           gt_volume_value=gt_volume_value)
        return losses


    def simple_test(self, points, volume_query_points, surface_query_points):
        nocs_results = self.garmentnet_nocs.simple_test(points)
        in_feature_volume = self.volume_agg(nocs_results)
        unet3d_result = self.unet3d_forward(in_feature_volume)
        volume_decoder_result = self.volume_decoder_forward(
            unet3d_result, volume_query_points)  # (volume_query_points 2, 6000, 3)
        surface_decoder_result = self.surface_decoder_forward(
            unet3d_result, surface_query_points)
        result = {
            'pointnet2_result': nocs_results,
            'unet3d_result': unet3d_result,
            'volume_decoder_result': volume_decoder_result,
            'surface_decoder_result': surface_decoder_result
        }
        return result

    def forward_test(self, points, volume_query_points, surface_query_points):
        nocs_results = self.garmentnet_nocs.simple_test(points)
        in_feature_volume = self.volume_agg(nocs_results)
        unet3d_result = self.unet3d_forward(in_feature_volume)
        volume_decoder_result = self.volume_decoder_forward(
            unet3d_result, volume_query_points)  # (volume_query_points 2, 6000, 3)
        surface_decoder_result = self.surface_decoder_forward(
            unet3d_result, surface_query_points)
        result = {
            'pointnet2_result': nocs_results,
            'unet3d_result': unet3d_result,
            'volume_decoder_result': volume_decoder_result,
            'surface_decoder_result': surface_decoder_result
        }
        return result


    def volume_decoder_forward(self, unet3d_result, query_points):
        out_feature_volume = unet3d_result['out_feature_volume']
        out_features = self.volume_decoder(out_feature_volume, query_points)
        pred_volume_value = out_features.view(*out_features.shape[:-1]) # (2, 6000, 1)
        decoder_result = {
            'out_features': out_features,
            'pred_volume_value': pred_volume_value
        }
        return decoder_result


    def unet3d_forward(self,
                       xyz,
                       pred_nocs,
                       per_point_features,
                       pred_confidence,
                       batch_size,
                       batch_idx
                       ):
        # volume agg
        in_feature_volume = self.volume_agg(
            xyz,
            pred_nocs,
            per_point_features,
            pred_confidence,
            batch_size,
            batch_idx
        )
        # unet3d
        out_feature_volume = self.unet_3d(in_feature_volume)
        unet3d_result = {
            'out_feature_volume': out_feature_volume
        }
        return unet3d_result

    def surface_decoder_forward(self, unet3d_result, query_points):
        out_feature_volume = unet3d_result['out_feature_volume']
        out_features = self.surface_decoder(out_feature_volume, query_points)
        decoder_result = {
            'out_features': out_features
        }
        return decoder_result

    def loss(self,
             pred_volume_value,
             pred_sim_points,
             gt_volume_value,
             gt_sim_points):
        losses = dict()
        loss_volume = self.volume_loss_weight * self.criterion(pred_volume_value, gt_volume_value)
        loss_surface = self.surface_loss_weight * self.criterion(pred_sim_points, gt_sim_points)
        losses['loss_volume'] = loss_volume
        losses['loss_surface'] = loss_surface
        return losses

    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def extract_feat(self, imgs):
        pass

@HEADS.register_module()
class VolumeFeatureAggregator(nn.Module):
    def __init__(self,
                 nn_channels=[1024, 1024, 128],
                 batch_norm=True,
                 lower_corner=(0, 0, 0),
                 upper_corner=(1, 1, 1),
                 grid_shape=(32, 32, 32),
                 reduce_method='mean',
                 include_point_feature=True,
                 include_confidence_feature=False):
        super().__init__()
        self.local_nn = MLP(nn_channels, batch_norm=batch_norm)
        self.lower_corner = tuple(lower_corner)
        self.upper_corner = tuple(upper_corner)
        self.grid_shape = tuple(grid_shape)
        self.reduce_method = reduce_method
        self.include_point_feature = include_point_feature
        self.include_confidence_feature = include_confidence_feature

    def forward(self, xyz, pred_nocs, per_point_features, pred_confidence,
                batch_size, batch_idx
                ):
        local_nn = self.local_nn
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        include_point_feature = self.include_point_feature
        include_confidence_feature = self.include_confidence_feature
        reduce_method = self.reduce_method

        # batch_size = points.shape[0]
        sim_points = xyz
        points = pred_nocs
        nocs_features = per_point_features
        confidence = pred_confidence


        device = points.device
        float_dtype = points.dtype
        int_dtype = torch.int64

        vg = VirtualGrid(
            lower_corner=lower_corner,
            upper_corner=upper_corner,
            grid_shape=grid_shape,
            batch_size=batch_size,
            device=device,
            int_dtype=int_dtype,
            float_dtype=float_dtype)

        # get aggregation target index
        points_grid_idxs = vg.get_points_grid_idxs(points, batch_idx=batch_idx)  # points: 第一阶段预测出的nocs (n, 3)
        flat_idxs = vg.flatten_idxs(points_grid_idxs, keepdim=False)

        # get features
        features_list = [nocs_features]
        if include_point_feature:
            points_grid_points = vg.idxs_to_points(points_grid_idxs)
            local_offset = points - points_grid_points
            features_list.append(local_offset)
            features_list.append(sim_points)

        if include_confidence_feature:
            features_list.append(confidence)
        features = torch.cat(features_list, axis=-1)

        # per-point transform
        if local_nn is not None:
            features = local_nn(features)

        # scatter
        volume_feature_flat = torch_scatter.scatter(
            src=features.T, index=flat_idxs, dim=-1,
            dim_size=vg.num_grids, reduce=reduce_method)

        # reshape to volume
        feature_size = features.shape[-1]
        volume_feature = volume_feature_flat.reshape(
            (feature_size, batch_size) + grid_shape).permute((1, 0, 2, 3, 4))
        return volume_feature


class PointBatchNorm1D(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return super().forward(x.view(-1, x.shape[-1])).view(x.shape)

def MLP(channels, batch_norm=True):
    layers = list()
    for i in range(1, len(channels)):
        module_layers = [
            nn.Linear(channels[i - 1], channels[i]),
            nn.ReLU()]
        if batch_norm:
            module_layers.append(
                PointBatchNorm1D(channels[i]))
        module = nn.Sequential(*module_layers)
        layers.append(module)
    return nn.Sequential(*layers)

@HEADS.register_module()
class ImplicitWNFDecoder(nn.Module):
    def __init__(self, nn_channels=(128, 512, 512, 1),
                 batch_norm=True):
        super().__init__()
        self.mlp = MLP(nn_channels, batch_norm=batch_norm)

    def forward(self, features_grid, query_points):
        """
        features_grid: (N,C,D,H,W)
        query_points: (N,M,3)
        """
        # normalize query points to (-1, 1), which is
        # requried by grid_sample
        query_points_normalized = 2.0 * query_points - 1.0  # [0, 1] -> [-1, 1]
        # shape (N,C,M,1,1)
        sampled_features = F.grid_sample(
            input=features_grid,
            grid=query_points_normalized.view(
                *(query_points_normalized.shape[:2] + (1, 1, 3))),
            mode='bilinear', padding_mode='border',
            align_corners=True)
        # shape (N,M,C)
        sampled_features = sampled_features.view(
            sampled_features.shape[:3]).permute(0, 2, 1)

        # shape (N,M,C)
        out_features = self.mlp(sampled_features)
        return out_features
