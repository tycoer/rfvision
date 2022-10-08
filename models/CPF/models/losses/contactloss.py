import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.CPF.datasets.hoquery import TransQueries, BaseQueries, ContactQueries, CollateQueries
from .bases.focal_loss import sigmoid_focal_loss, multiclass_focal_loss
from mmdet3d.models.builder import LOSSES

@LOSSES.register_module()
class VertexContactLoss:
    def __init__(
        self,
        contact_lambda_vertex_contact=1.0,
        contact_lambda_contact_region=1.0,
        contact_lambda_anchor_elasti=1.0,
        # sample_balance_multiplier=9.0,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2,
        region_focal_loss_alpha=None,
    ):
        self.contact_lambda_contact_region = contact_lambda_contact_region
        self.contact_lambda_anchor_elasti = contact_lambda_anchor_elasti
        # self.sample_balance_multiplier = sample_balance_multiplier
        self.contact_lambda_vertex_contact = contact_lambda_vertex_contact
        # ! for focal loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

        # ! for region focal loss
        if region_focal_loss_alpha == "fhb":
            self.region_focal_loss_alpha = torch.Tensor(
                [
                    0.09948797,
                    0.02393614,
                    0.00958673,
                    0.03823604,
                    0.02974173,
                    0.01377778,
                    0.19825081,
                    0.0638811,
                    0.03167253,
                    0.10425754,
                    0.08043329,
                    0.02713633,
                    0.0742536,
                    0.08478072,
                    0.03479149,
                    0.02330341,
                    0.0624728,
                ]
            )
        elif region_focal_loss_alpha == "ho3d":
            self.region_focal_loss_alpha = torch.Tensor(
                [
                    0.10156912,
                    0.02822553,
                    0.01931505,
                    0.04377314,
                    0.02218014,
                    0.01778827,
                    0.19733622,
                    0.03988811,
                    0.04025808,
                    0.19438567,
                    0.03801038,
                    0.02245348,
                    0.07134447,
                    0.05416137,
                    0.02648691,
                    0.01691766,
                    0.06590642,
                ]
            )
        elif torch.is_tensor(region_focal_loss_alpha):
            self.region_focal_loss_alpha = region_focal_loss_alpha
        else:
            self.region_focal_loss_alpha = None

    def __call__(self, preds, targs):
        final_loss = None

        contact_losses = {}

        # * =============================== VERTEX CONTACT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.contact_lambda_vertex_contact and ContactQueries.VERTEX_CONTACT in targs:
            recov_vertex_contact = preds["recov_vertex_contact"]
            target_device = recov_vertex_contact.device
            vertex_contact_gt = targs[ContactQueries.VERTEX_CONTACT].float().to(target_device)

            # ? 1. we need to filter out the points that lie outside the image
            recov_contact_in_image_mask = preds["recov_contact_in_image_mask"]  # TENSOR (B, N)

            #! ATTENTION! BUGGY if you directly apply padding_mask on vertex_contact
            # // masked_pred = recov_vertex_contact * recov_contact_in_image_mask
            # // masked_gt = vertex_contact_gt * recov_contact_in_image_mask

            # ? 2. also, we need to filter out the points introduced in collate
            # collate_padding_mask = targs[CollateQueries.PADDING_MASK].float()  # TENSOR (B, N)
            # collate_padding_mask = collate_padding_mask.to(target_device)
            #! ATTENTION! BUGGY if you directly apply padding_mask on vertex_contact
            # // masked_pred = masked_pred * collate_padding_mask
            # // masked_gt = masked_gt * collate_padding_mask
            # combined_mask = recov_contact_in_image_mask * collate_padding_mask  # TENSOR (B, N)
            combined_mask = recov_contact_in_image_mask

            vertex_contact_loss = sigmoid_focal_loss(
                inputs=recov_vertex_contact,
                targets=vertex_contact_gt,
                masks=combined_mask,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="mean",
            )
            if final_loss is None:
                final_loss = torch.Tensor([0.0]).float().to(target_device)
            final_loss += self.contact_lambda_vertex_contact * vertex_contact_loss
            contact_losses["recov_vertex_contact"] = vertex_contact_loss
        # * <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # * =============================== CONTACT REGION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.contact_lambda_contact_region and ContactQueries.CONTACT_REGION_ID in targs:
            recov_contact_region = preds["recov_contact_region"]  # TENSOR(B, N, C)
            target_device = recov_contact_region.device
            n_regions = recov_contact_region.shape[2]  # C: 17
            recov_contact_region = recov_contact_region.view((-1, n_regions))  # TENSOR (BxN, C)

            # ======== convert gt region idx to one-hot >>>>>>>>>>>>
            contact_region_gt = targs[ContactQueries.CONTACT_REGION_ID]  # TENSOR(B, N)
            contact_region_gt = contact_region_gt.view((-1, 1))  # TENSOR(BxN, 1)
            contact_region_idx = contact_region_gt.long()

            # TENSOR(BxN, C+1)
            # considering vertexs without contact as background
            # the background class has the largest index (eg. 17)
            contact_region_one_hot_with_back = torch.FloatTensor(contact_region_gt.shape[0], n_regions + 1).zero_().to(contact_region_idx.device)
            contact_region_one_hot_with_back = contact_region_one_hot_with_back.scatter_(1, contact_region_idx, 1)

            contact_region_one_hot = contact_region_one_hot_with_back[:, :n_regions]  # TENSOR (BxN , C)
            contact_region_one_hot = contact_region_one_hot.to(target_device)

            # ============== construct the mask >>>>>>>>>>>>>>>>>
            # ? 1. we need to filter out the points that lie outside the image
            recov_contact_in_image_mask = preds["recov_contact_in_image_mask"].view((-1, 1))  # TENSOR (BxN, 1)

            # ? 2. also, we need to filter out the points introduced in collate
            # collate_padding_mask = targs[CollateQueries.PADDING_MASK].float().view((-1, 1))  # TENSOR (BxN, 1)
            # collate_padding_mask = collate_padding_mask.to(target_device)

            # ? 3. third, we need to filter out the non-contact points in gt
            contact_filtering_mask = (contact_region_gt != n_regions).float()  # TENSOR (BxN, 1)
            contact_filtering_mask = contact_filtering_mask.to(target_device)

            region_combined_mask = (
                # recov_contact_in_image_mask * collate_padding_mask * contact_filtering_mask
                recov_contact_in_image_mask * contact_filtering_mask

            )  # TENSOR (BxN, 1)

            contact_region_loss = multiclass_focal_loss(
                inputs=recov_contact_region,
                targets=contact_region_one_hot,
                masks=region_combined_mask,
                alpha=self.region_focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="mean",
            )
            if final_loss is None:
                final_loss = torch.Tensor([0.0]).float().to(target_device)
            final_loss += self.contact_lambda_contact_region * contact_region_loss
            contact_losses["recov_contact_region"] = contact_region_loss
        # * <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # * =============================== ANCHOR ELASTI >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if (
            self.contact_lambda_contact_region
            and ContactQueries.CONTACT_REGION_ID in targs
            and self.contact_lambda_anchor_elasti
            and ContactQueries.CONTACT_ANCHOR_ELASTI in targs
        ):
            recov_anchor_elasti = preds["recov_anchor_elasti"]  # TENSOR (B, N, 4)
            target_device = recov_anchor_elasti.device
            anchor_elasti_gt = targs[ContactQueries.CONTACT_ANCHOR_ELASTI].float()  # TENSOR (B, N, 4)
            anchor_elasti_gt = anchor_elasti_gt.to(target_device)

            maximum_anchor = recov_anchor_elasti.shape[2]  # 4

            recov_anchor_elasti = recov_anchor_elasti.view((-1, maximum_anchor))  # TENSOR (BxN, 4)
            anchor_elasti_gt = anchor_elasti_gt.view((-1, maximum_anchor))  # TENSOR (BxN, 4)

            # ============== construct the mask >>>>>>>>>>>>>>>>>
            # ? 1. filter out the points that lie outside the image
            # ? 2. filter out the points introduced in collate
            # ? 3. filter out the non-contact points in gt
            # ? NOTE: 1.2.3. is already done as:  region_combined_mask
            region_combined_mask = region_combined_mask.repeat(1, maximum_anchor)  # TENSOR (BxN, 4)

            # ? 4. filter out the unbalanced region-anchors by CONTACT_ANCHOR_PADDING_MASK
            anchor_padding_mask = targs[ContactQueries.CONTACT_ANCHOR_PADDING_MASK].float()  # TENSOR (B, N, 4)
            anchor_padding_mask = anchor_padding_mask.to(target_device)
            anchor_padding_mask = anchor_padding_mask.view((-1, maximum_anchor))  # TENSOR (BxN, 4)

            anchor_combined_mask = anchor_padding_mask * region_combined_mask  # TENSOR (BxN, 4)
            anchor_combined_mask.requires_grad_ = False

            # *: bce @lixin finished
            # // anchor_elasti_loss = F.l1_loss(
            # //    input=recov_anchor_elasti * anchor_combined_mask,
            # //    target=anchor_elasti_        self.contact_lambda_vertex_contact = contact_lambda_vertex_contactgt * anchor_combined_mask,
            # //    reduction="sum",
            # //)
            if anchor_combined_mask.sum().detach().cpu().item() != 0:
                anchor_elasti_loss = F.binary_cross_entropy(
                    input=recov_anchor_elasti, target=anchor_elasti_gt, reduction="none"
                )  # TENSOR (BxN, 4)
                anchor_elasti_loss = (anchor_elasti_loss * anchor_combined_mask).sum()  # TENSOR (BxN, 4) -> SUM
                anchor_elasti_loss = anchor_elasti_loss / (anchor_combined_mask.sum())  # reduction mean
                if final_loss is None:
                    final_loss = torch.Tensor([0.0]).float().to(target_device)
                final_loss += self.contact_lambda_anchor_elasti * anchor_elasti_loss
                contact_losses["recov_anchor_elasti"] = anchor_elasti_loss
            else:
                anchor_elasti_loss = torch.Tensor([0.0]).float().to(target_device)
                final_loss += torch.Tensor([0.0]).float().to(target_device)
                contact_losses["recov_anchor_elasti"] = anchor_elasti_loss

        assert final_loss is not None, "no criterion is computed in contact loss"
        contact_losses["contact_info_total_loss"] = final_loss
        return final_loss, contact_losses
