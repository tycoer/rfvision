from models.CAPTRA.models.captra_trackmodel import CaptraTrackModel
from models.CAPTRA.test.test_coordnet import obj_cfg, data

obj_category='1'


if __name__ == '__main__':
    cfg = dict(
        obj_cfg=obj_cfg[obj_category],
        rotnet=dict(
            nocs_head_dims=[128],
            obj_cfg = obj_cfg[obj_category],
            network_type = 'rot',
            backbone_out_dim = 128,
        ),
        coordnet=dict(
            obj_cfg=obj_cfg[obj_category],
            backbone_out_dim=128,
            extra_dims=1,
        ),
        track_cfg=dict(gt_label=False,
                       nocs2d_label=False,
                       nocs2d_path=None),
        init_frame=dict(gt=False),
        nocs_otf=True,
        batch_size=1,
        root_dset='/disk1/data/captar',
        obj_category='1',
        data_radius=0.6,
        pose_perturb_cfg=dict()
    )

    m = CaptraTrackModel(cfg)

