model=dict(
    type='GarmentnetNOCS',
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=6,
        num_points=(4096, 2048, 1024),
        radius=(0.05, 0.1, 0.2),
        num_samples=(64, 32, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     ),
        fp_channels=((256, 256), (256, 256)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True),
        init_cfg=None
    ),
    head=dict(
        type='GarmentnetNOCSHead',
        nocs_bins=64,
        dropout=True,
        feature_dim=128,
        nocs_loss_weight=1,
        grip_point_loss_weight=1,
    )
)

data_cfg = dict(
    zarr_path='data/garmentnets_dataset_sample.zarr/Tshirt',  # data root
    metadata_cache_dir='~/local/.cache/metadata_cache_dir',
    # sample size
    num_pc_sample=4096,
    num_volume_sample=0,
    num_surface_sample=0,
    num_mc_surface_sample=0,
    # mixed sampling config
    surface_sample_ratio=0,
    surface_sample_std=0.05,
    # surface sample noise
    surface_normal_noise_ratio=0,
    surface_normal_std=0,
    # data augumentaiton
    enable_augumentation=True,
    random_rot_range=(-90, 90),
    num_views=4,
    pc_noise_std=0,
    # volume config
    volume_size=128,
    volume_group='nocs_winding_number_field',
    tsdf_clip_value=None,
    volume_absolute_value=False,
    include_volume=False,
    # random seed
    static_epoch_seed=False,
)

train_pipeline = [
    dict(type='Collect3D', keys=['points', 'gt_nocs', 'gt_grip_point'])]
test_pipeline = [dict(type='Collect3D', keys=['points'])]
test_pipeline = [dict(type='Collect3D', keys=['points'])]


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(type='GarmentnetDataset', pipeline=train_pipeline, **data_cfg),
    val=dict(type='GarmentnetDataset', pipeline=test_pipeline, **data_cfg),
    test=dict(type='GarmentnetDataset', pipeline=test_pipeline, **data_cfg))



optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=9000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict()

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
