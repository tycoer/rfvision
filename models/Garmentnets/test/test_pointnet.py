from mmdet3d.models.backbones.pointnet2_sa_ssg import PointNet2SASSG
import torch

cfg=dict(
    # type='PointNet2SASSG',
    in_channels=6,
    num_points=(8192, 4096, 2048),
    radius=(0.05, 0.1, 0.1),
    num_samples=(64, 32, 16),
    sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256)),
    fp_channels=((256, 256), (256, 256)),
    norm_cfg=dict(type='BN2d'),
    sa_cfg=dict(
        type='PointSAModule',
        pool_mod='max',
        use_xyz=True,
        normalize_xyz=True),
    init_cfg=None
)

if __name__ == '__main__':
    points = torch.rand(2, 6000, 6).cuda()
    m = PointNet2SASSG(**cfg).cuda()
    res = m(points)

    for i in res['fp_features']:
        print(i.shape)

    m.eval()
