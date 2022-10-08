import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_f
from models.CPF.datasets.hoquery import TransQueries
from mmdet3d.models.builder import LOSSES

@LOSSES.register_module()
class ManoLoss:
    def __init__(
        self,
        mano_lambda_recov_joints3d: float = 0.5,
        mano_lambda_recov_verts3d: float = 0.5,
        lambda_verts3d: float = None,
        lambda_joints3d: float = None,
        lambda_shape: float = None,
        lambda_pose_reg: float = None,
    ):
        """
        Computed terms of MANO weighted loss, which encompasses vertex/joint
        supervision and pose/shape regularization
        """
        self.mano_lambda_recov_joints3d = mano_lambda_recov_joints3d
        self.mano_lambda_recov_verts3d = mano_lambda_recov_verts3d
        self.lambda_verts3d = lambda_verts3d
        self.lambda_joints3d = lambda_joints3d
        self.lambda_shape = lambda_shape
        self.lambda_pose_reg = lambda_pose_reg

    def __call__(self, preds, targs):
        final_loss = None
        reg_loss = None
        mano_losses = {}

        # If needed, compute and add vertex loss
        if TransQueries.HAND_VERTS_3D in targs and self.lambda_verts3d:
            # both root relative
            pred_verts3d = preds["verts3d"]
            target_device = pred_verts3d.device
            targ_verts3d = (targs[TransQueries.HAND_VERTS_3D] - targs[TransQueries.CENTER_3D].unsqueeze(1)).to(
                target_device
            )
            verts3d_loss = torch_f.mse_loss(pred_verts3d, targ_verts3d)

            if final_loss is None:
                final_loss = torch.Tensor([0.0]).float().to(target_device)
            final_loss += self.lambda_verts3d * verts3d_loss
        else:
            verts3d_loss = None
        mano_losses["mano_verts3d"] = verts3d_loss

        # Compute joints loss in all cases
        if TransQueries.JOINTS_3D in targs and self.lambda_joints3d:
            pred_joints3d = preds["joints3d"]
            target_device = pred_joints3d.device
            targ_joints3d = (targs[TransQueries.JOINTS_3D] - targs[TransQueries.CENTER_3D].unsqueeze(1)).to(
                target_device
            )

            # Add to final_loss for backpropagation if needed
            joints3d_loss = torch_f.mse_loss(pred_joints3d, targ_joints3d)
            if final_loss is None:
                final_loss = torch.Tensor([0.0]).float().to(target_device)
            final_loss += self.lambda_joints3d * joints3d_loss
        else:
            joints3d_loss = None
        mano_losses["mano_joints3d"] = joints3d_loss

        # Compute hand shape regularization loss
        if self.lambda_shape:
            pred_shape = preds["shape"]
            target_device = pred_shape.device
            shape_reg_loss = torch_f.mse_loss(pred_shape, torch.zeros_like(pred_shape))
            if final_loss is None:
                final_loss = torch.Tensor([0.0]).float().to(target_device)
            if reg_loss is None:
                reg_loss = torch.Tensor([0.0]).float().to(target_device)
            final_loss += self.lambda_shape * shape_reg_loss
            reg_loss += self.lambda_shape * shape_reg_loss
        else:
            shape_reg_loss = None
        mano_losses["mano_shape"] = shape_reg_loss

        # Compute hand pose regularization loss
        if self.lambda_pose_reg:
            pred_pose = preds["pose"][:, 3:]  # ignore root rotations at [:, :3]
            target_device = pred_pose.device
            pose_reg_loss = torch_f.mse_loss(pred_pose, torch.zeros_like(pred_pose))
            if final_loss is None:
                final_loss = torch.Tensor([0.0]).float().to(target_device)
            if reg_loss is None:
                reg_loss = torch.Tensor([0.0]).float().to(target_device)
            final_loss += self.lambda_pose_reg * pose_reg_loss
            reg_loss += self.lambda_pose_reg * pose_reg_loss
        else:
            pose_reg_loss = None
        mano_losses["pose_reg"] = pose_reg_loss

        # Compute hand losses in and camera coordinates
        # ! CHANGE FROM BaseQueries TO TransQueries
        if self.mano_lambda_recov_joints3d and TransQueries.JOINTS_3D in targs:
            recov_joints3d = preds["recov_joints3d"]
            target_device = recov_joints3d.device
            joints3d_gt = targs[TransQueries.JOINTS_3D].to(target_device)
            recov_joints3d_loss = torch_f.mse_loss(recov_joints3d, joints3d_gt)
            if final_loss is None:
                final_loss = torch.Tensor([0.0]).float().to(target_device)
            final_loss += self.mano_lambda_recov_joints3d * recov_joints3d_loss
        else:
            recov_joints3d_loss = None
        mano_losses["recov_joints3d"] = recov_joints3d_loss

        # ! CHANGE FROM BaseQueries TO TransQueries
        if self.mano_lambda_recov_verts3d and TransQueries.HAND_VERTS_3D in targs:
            recov_hand_verts3d = preds["recov_hand_verts3d"]
            target_device = recov_hand_verts3d.device
            hand_verts3d_gt = targs[TransQueries.HAND_VERTS_3D].to(target_device)

            recov_hand_verts3d_loss = torch_f.mse_loss(recov_hand_verts3d, hand_verts3d_gt)
            if final_loss is None:
                final_loss = torch.Tensor([0.0]).float().to(target_device)
            final_loss += self.mano_lambda_recov_verts3d * recov_hand_verts3d_loss
        else:
            recov_hand_verts3d_loss = None
        mano_losses["recov_hand_verts3d"] = recov_hand_verts3d_loss

        mano_losses["mano_total_loss"] = final_loss
        mano_losses["mano_reg_loss"] = reg_loss
        return final_loss, mano_losses
