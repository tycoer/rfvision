from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.models.builder import DETECTORS, build_backbone, build_loss
import torch.nn as nn
from ..utils.eval import Evaluator
from ..utils.lossutils import update_loss
from ..datasets.hoquery import recover_queries
import torch

@DETECTORS.register_module()
class CPFPicrOffline(Base3DDetector):
    def __init__(self,
                 backbone,
                 loss_contact,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.loss_contact = build_loss(loss_contact)
        if self.training:
            self.evaluator = Evaluator()

    def forward_train(self, img_metas, **sample):
        sample = recover_queries(sample)
        results = self.backbone(sample)
        loss_contact = 0
        infos = []
        for result in results:
            loss_contact_, info = self.loss_contact(result, sample)
            loss_contact +=  loss_contact_
            infos.append(info)

        infos = update_loss(infos)
        for loss_name, loss_val in infos.items():
            if loss_val is not None:
                self.evaluator.add_loss_value(loss_name, loss_val.mean().item())

        self.evaluator.feed_loss_meters(sample, results[-1])
        self.evaluator.feed_eval_meters(sample, results[-1])
        # self.evaluator.loss_meters["contact_info_total_loss"].avg * 1.0,
        acc = self.evaluator.eval_meters["vertex_contact"].acc * 1.0,
        pc = self.evaluator.eval_meters["vertex_contact"].pc * 1.0,
        rc = self.evaluator.eval_meters["vertex_contact"].rc * 1.0,
        f1 = self.evaluator.eval_meters["vertex_contact"].f1 * 1.0,
        mre = self.evaluator.eval_meters["contact_region"].acc * 1.0,
        ae = self.evaluator.loss_meters["recov_anchor_elasti"].avg * 1.0,


        losses = dict(loss_contact=loss_contact,
                      acc=torch.tensor(acc, dtype=torch.float32),
                      pc=torch.tensor(pc, dtype=torch.float32),
                      rc=torch.tensor(rc, dtype=torch.float32),
                      f1=torch.tensor(f1, dtype=torch.float32),
                      mre=torch.tensor(mre, dtype=torch.float32),
                      ae=torch.tensor(ae, dtype=torch.float32),
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