model=dict(
    type='GarmentnetWNF',
    backbone=dict(
        type='UNet3D',
        in_channels=128,
        out_channels=128,
        f_maps=32,
        layer_order='gcr',
        num_groups=8,
        num_levels=4,
    ),
    garmentnet_nocs=dict(
        type='GarmentnetNOCS',
        backbone=dict(
            type='PointNet2SASSG',
            in_channels=6,
            num_points=(8192, 4096, 2048, 1024),
            radius=(0.2, 0.4, 0.8, 1.2),
            num_samples=(64, 32, 16, 16),
            sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                         (128, 128, 256)),
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
        ),
    ),
    volume_agg_params=dict(
        type='VolumeFeatureAggregator',
        nn_channels = [137, 137, 128] ,
        batch_norm = True ,
        lower_corner = [0, 0, 0] ,
        upper_corner = [1, 1, 1] ,
        grid_shape = [32, 32, 32] ,
        reduce_method = 'max' ,
        include_point_feature = True ,
        include_confidence_feature = True ,
    ),
    volume_decoder_params=dict(
        type='ImplicitWNFDecoder',
        nn_channels=[128, 256, 256, 1],
        batch_norm=True,
    ),
    surface_decoder_params=dict(
        type='ImplicitWNFDecoder',
        nn_channels=[128, 256, 256, 3] ,
        batch_norm=True ,
    ),
    volume_loss_weight=1.0,
    surface_loss_weight=1.0,
)

data_cfg = dict(
    zarr_path='/home/hanyang/garmentnets-master/data/garmentnets_dataset_sample.zarr/Tshirt',  # data root
    metadata_cache_dir='~/local/.cache/metadata_cache_dir',
    # sample size
    batch_size=24,
    num_workers=0,
    # sample size
    num_pc_sample=4096,
    num_volume_sample=4096,
    num_surface_sample=4096,
    num_mc_surface_sample=0,
    # mixed sampling config
    surface_sample_ratio=0,
    surface_sample_std=0.05,
    # surface sample noise
    # use 0.5
    surface_normal_noise_ratio=0,
    surface_normal_std=0.01,
    # data augumentation
    enable_augumentation=True,
    random_rot_range=[-180, 180],
    num_views=4,
    # volume
    volume_size=128,
    # or nocs_signed_distance_field or nocs_occupancy_grid or sim_nocs_winding_number_field or nocs_distance_field
    volume_group='nocs_winding_number_field',
    # use 0.05
    tsdf_clip_value=None,
    volume_absolute_value=False,
    include_volume=False,
    # random seed
    static_epoch_seed=False,
    # datamodule config
    dataset_split=[8, 1, 1],
    split_seed=0
)

train_pipeline = [
    dict(type='Collect3D', keys=['points', 'volume_query_points', 'surface_query_points', 'gt_volume_value', 'gt_sim_points'])]
test_pipeline = [dict(type='Collect3D', keys=['points'])]
test_pipeline = [dict(type='Collect3D', keys=['points'])]


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(type='GarmentnetDataset', pipeline=train_pipeline, **data_cfg),
    val=dict(type='GarmentnetDataset', pipeline=test_pipeline, **data_cfg),
    test=dict(type='GarmentnetDataset', pipeline=test_pipeline, **data_cfg))



optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)


log_config = dict(
    interval=1,
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
