from models.CAPTRA.models.captra_backbone import CoordNet, PartCanonNet

if __name__ == '__main__':
    cfg = dict(network=dict(backbone_out_dim=128, nocs_head_dims=[128],),
               obj=dict(extra_dims=1,),
               num_parts=1,
               obj_sym=True,
               pointnet=dict(in_channels=3,
                             num_points=(512, 128),
                             radii=((0.05, 0.1, 0.2), (0.2, 0.4)),
                             num_samples=((32, 64, 128), (64, 128)),
                             sa_channels=[[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                                          [[128, 128, 256], [128, 196, 256]],
                                          ],
                             aggregation_channels=(64, 128),
                             out_indices=(-1,),
                             )
               )

    cfg = dict()

    m = CoordNet(**cfg)
