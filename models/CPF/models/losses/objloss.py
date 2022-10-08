import torch
import torch.nn.functional as torch_f

from models.CPF.datasets.hoquery import TransQueries
from mmdet3d.models.builder import LOSSES

def normalize_pixel_out(data, inp_res):
    batch_size = data.shape[0]
    img_centers = (data.new(inp_res) / 2).view(1, 2).repeat(batch_size, 1).unsqueeze(1)
    centered = data - img_centers
    scaled = centered / img_centers
    return scaled.float()

@LOSSES.register_module()
class ObjLoss:
    def __init__(self, obj_lambda_recov_verts3d=0.5, obj_lambda_recov_verts2d=0.5):
        self.obj_lambda_recov_verts3d = obj_lambda_recov_verts3d
        self.obj_lambda_recov_verts2d = obj_lambda_recov_verts2d

    def __call__(self, preds, targs):
        final_loss = None
        obj_losses = {}

        if self.obj_lambda_recov_verts3d and TransQueries.OBJ_VERTS_3D in targs:
            recov_obj_verts3d = preds["recov_obj_verts3d"]
            target_device = recov_obj_verts3d.device
            obj_verts3d_gt = targs[TransQueries.OBJ_VERTS_3D].to(target_device)
            recov_obj_verts3d_loss = torch_f.mse_loss(recov_obj_verts3d, obj_verts3d_gt)
            if final_loss is None:
                final_loss = torch.Tensor([0.0]).float().to(target_device)
            final_loss += self.obj_lambda_recov_verts3d * recov_obj_verts3d_loss
        else:
            recov_obj_verts3d_loss = None
        obj_losses["recov_obj_verts3d"] = recov_obj_verts3d_loss

        if self.obj_lambda_recov_verts2d and TransQueries.OBJ_VERTS_2D in targs:
            pred_obj_verts2d = preds["obj_verts2d"]
            target_device = pred_obj_verts2d.device
            obj_verts2d_gt = targs[TransQueries.OBJ_VERTS_2D].to(target_device)
            height, width = tuple(targs[TransQueries.IMAGE].shape[2:])

            recov_obj_verts2d_loss = torch_f.smooth_l1_loss(
                normalize_pixel_out(pred_obj_verts2d, inp_res=(width, height)),
                normalize_pixel_out(obj_verts2d_gt, inp_res=(width, height)),
            )

            if final_loss is None:
                final_loss = torch.Tensor([0.0]).float().to(target_device)
            final_loss += self.obj_lambda_recov_verts2d * recov_obj_verts2d_loss
        else:
            recov_obj_verts2d_loss = None
        obj_losses["recov_obj_verts2d"] = recov_obj_verts2d_loss

        obj_losses["obj_total_loss"] = final_loss
        return final_loss, obj_losses

