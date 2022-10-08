from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.models.builder import DETECTORS, build_backbone, build_loss
import torch.nn as nn
from ..utils.eval import Evaluator
from ..datasets.hoquery import recover_queries
import torch

@DETECTORS.register_module()
class CPFHoNet(Base3DDetector):
    def __init__(self,
                 backbone,
                 loss_mano,
                 loss_obj,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.loss_mano = build_loss(loss_mano)
        self.loss_obj = build_loss(loss_obj)
        if self.training:
            self.evaluator = Evaluator()

    def forward_train(self, img_metas, **sample):
        sample = recover_queries(sample)
        results = self.backbone(sample)
        loss_mano, _ = self.loss_mano(results, sample)
        loss_obj, _ = self.loss_obj(results, sample)

        self.evaluator.feed_loss_meters(sample, results)
        self.evaluator.feed_eval_meters(sample, results)
        recov_obj_verts3d_mepe = self.evaluator.loss_meters["recov_obj_verts3d_mepe"].avg * 1000.0
        recov_hand_verts3d_mepe = self.evaluator.loss_meters["recov_hand_verts3d_mepe"].avg * 1000.0
        recov_joints3d_mepe = self.evaluator.loss_meters["recov_joints3d_mepe"].avg * 1000.0


        losses = dict(loss_mano=loss_mano,
                      loss_obj=loss_obj,
                      recov_joints3d_mepe=torch.tensor(recov_joints3d_mepe, dtype=torch.float32),
                      recov_hand_verts3d_mepe=torch.tensor(recov_hand_verts3d_mepe, dtype=torch.float32),
                      recov_obj_verts3d_mepe=torch.tensor(recov_obj_verts3d_mepe, dtype=torch.float32)
                      )

        return losses

    def forward_test(self, img_metas, rescale=True, **sample):
        sample = recover_queries(sample)
        results = self.backbone(sample)
        results = [results]
        return results

    def simple_test(self, img_metas, rescale=True, **sample):
        return self.forward_test(img_metas, **sample)

    def extract_feat(self, imgs):
        pass
    def aug_test(self, imgs, img_metas, **kwargs):
        pass




if __name__ == '__main__':
    pass