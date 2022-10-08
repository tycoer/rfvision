from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors import Base3DDetector
from .asdf_decoder import Decoder
import torch
import torch.nn.functional as F

@DETECTORS.register_module()
class ASDF(Base3DDetector):
    def __init__(self,
                 # decoder para
                 latent_size=253,
                 dims=[512, 512, 512, 768, 512, 512, 512, 512],
                 dropout=[0, 1, 2, 3, 4, 5, 6, 7],
                 dropout_prob=0.2,
                 norm_layers=[0, 1, 2, 3, 4, 5, 6, 7],
                 latent_in=[4],
                 articulation=True,  # for development
                 num_atc_parts=1,
                 do_sup_with_part=False,
                 init_cfg=None,

                 # train cfg
                 code_bound=1.0,
                 num_scenes=92,
                 do_code_regularization=True,
                 num_samp_per_scene=16000,
                 enforce_minmax=True,
                 batch_split=1,
                 clamp_dist=0.1,
                 code_reg_lambda=0.0001,

                 **kwargs
                 ):
        super().__init__(init_cfg)
        self.decoder = Decoder(latent_size=latent_size,
                               dims=dims,
                               dropout=dropout,
                               dropout_prob=dropout_prob,
                               norm_layers=norm_layers,
                               latent_in=latent_in,
                               articulation=articulation,
                               num_atc_parts=num_atc_parts,
                               do_sup_with_part=do_sup_with_part,
                               # init_cfg=init_cfg
                               )

        self.articulation = articulation
        self.do_sup_with_part = do_sup_with_part
        self.do_code_regularization = do_code_regularization
        self.num_samp_per_scene = num_samp_per_scene
        self.enforce_minmax = enforce_minmax
        self.clamp_dist = clamp_dist
        self.batch_split = batch_split
        self.num_atc_parts = num_atc_parts
        self.code_reg_lambda = code_reg_lambda
        self.minT, self.maxT = -self.clamp_dist, self.clamp_dist


        self.lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
        self.loss_l1 = torch.nn.L1Loss(reduction='sum')

    def forward_train(self, img_metas, **kwargs):
        # tycoer
        losses = dict()
        all_sdf_data = [kwargs['sdf_data'], kwargs['atc'], kwargs['instance_idx']]
        indices = kwargs['indices']

        if self.articulation == True:
            sdf_data = all_sdf_data[0].reshape(-1, 5)
            atc = all_sdf_data[1].view(-1, self.num_atc_parts)
            instance_idx = all_sdf_data[2].view(-1, 1)
            atc = atc.repeat(1, all_sdf_data[0].size(1)).reshape(-1, self.num_atc_parts)
            instance_idx = instance_idx.repeat(1, all_sdf_data[0].size(1)).reshape(-1, 1)
            num_sdf_samples = sdf_data.shape[0]
            sdf_data[0].requires_grad = False
            sdf_data[1].requires_grad = False
            xyz = sdf_data[:, 0:3].float()
            sdf_gt = sdf_data[:, 3].unsqueeze(1)
            part_gt = sdf_data[:, 4].unsqueeze(1).long()
        else:
            sdf_data = all_sdf_data.reshape(-1, 5)
            num_sdf_samples = sdf_data.shape[0]
            sdf_data.requires_grad = False
            xyz = sdf_data[:, 0:3].float()
            sdf_gt = sdf_data[:, 3].unsqueeze(1)
            part_gt = sdf_data[:, 4].unsqueeze(1).long()

        xyz = torch.chunk(xyz, self.batch_split)
        indices = torch.chunk(
            indices.unsqueeze(-1).repeat(1, self.num_samp_per_scene).view(-1),
            self.batch_split,
        )

        if self.enforce_minmax:
            sdf_gt = torch.clamp(sdf_gt, self.minT, self.maxT)

        sdf_gt = torch.chunk(sdf_gt, self.batch_split)
        part_gt = torch.chunk(part_gt, self.batch_split)

        if self.articulation == True:
            atc = torch.chunk(atc, self.batch_split)
            instance_idx = torch.chunk(instance_idx, self.batch_split)

        for i in range(self.batch_split):

            if self.articulation == True:
                batch_vecs = self.lat_vecs(instance_idx[i].view(-1) - 1)
            else:
                batch_vecs = self.lat_vecs(indices[i])

            # NN optimization
            if self.articulation == True:
                input = torch.cat([batch_vecs, xyz[i], atc[i]], dim=1)
            else:
                input = torch.cat([batch_vecs, xyz[i]], dim=1)
            if self.do_sup_with_part:
                pred_sdf, pred_part = self.decoder(input)
            else:
                pred_sdf = self.decoder(input)

            if self.enforce_minmax:
                pred_sdf = torch.clamp(pred_sdf, self.minT, self.maxT)
            chunk_loss = self.loss_l1(pred_sdf, sdf_gt[i].cuda()) / num_sdf_samples
            losses['loss_chunk'] = chunk_loss
            if self.do_code_regularization:
                l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                reg_loss = (self.code_reg_lambda * l2_size_loss) / num_sdf_samples
                losses['loss_reg'] = reg_loss

            if self.do_sup_with_part:
                part_loss = F.cross_entropy(pred_part, part_gt[i].view(-1).cuda())
                part_loss *= 1e-3
                losses['loss_part'] = part_loss
        return losses

    @torch.no_grad()
    def forward_test(self, **kwargs):
        if self.do_sup_with_part:
            pred_sdf, pred_part = self.decoder(input)
            return pred_sdf, pred_part
        else:
            pred_sdf = self.decoder(input)
            return pred_sdf

    @torch.no_grad()
    def simple_test(self, img_metas, rescale=True, **sample):
        return self.forward_test(img_metas, **sample)

    def extract_feat(self, imgs):
        pass

    def aug_test(self, imgs, img_metas, **kwargs):
        pass




