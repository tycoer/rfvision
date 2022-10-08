assets_root="/home/hanyang/rfvision/models/CPF/assets"
data_root = "/hdd2/data/handata"
model_root = '/hdd2/data/handata/YCB_models'

model=dict(
    type='CPFHoNet',
    backbone=dict(
        type='HONet',
        fc_dropout=0,
        resnet_version=18,
        mano_neurons=[512, 512],
        mano_comps=15,
        mano_use_pca=True,
        mano_use_shape=True,
        mano_center_idx=9,
        assets_root=assets_root,
        mano_pose_coeff = 1,
        mano_fhb_hand = False,
        mano_lambda_recov_joints3d=0.5,
        mano_lambda_recov_verts3d=0,
        obj_lambda_recov_verts3d=0.5,
        obj_lambda_recov_verts2d=0,
        obj_trans_factor=100,
        obj_scale_factor=0.0001,),
    loss_mano=dict(
        type='ManoLoss',
        mano_lambda_recov_joints3d=0.5,
        mano_lambda_recov_verts3d=0,
        lambda_shape=5e-07,
        lambda_pose_reg=5e-06
    ),
    loss_obj=dict(
        type='ObjLoss',
        obj_lambda_recov_verts3d=0.5,
        obj_lambda_recov_verts2d=0),
)


data_cfg = dict(
    datasets='ho3d',
    data_root=data_root,
    data_split='train',
    split_mode='objects',
    use_cache=False,
    mini_factor=1,
    center_idx=9,
    enable_contact=False,
    filter_no_contact=False,
    filter_thresh=0.0,  # useless
    # query=query,
    like_v1=True,
    block_rot=True,
    max_rot=True,
    scale_jittering=0,
    center_jittering=0.1,
    synt_factor=(0, ),
    assets_root=assets_root,
    model_root=model_root,

    init_cfgs=dict(
        data_root=data_root,
        data_split="train",
        njoints=21,
        use_cache=True,
        filter_no_contact=False,
        filter_thresh=10.0,
        mini_factor=1.0,
        center_idx=9,
        scale_jittering=0.0,
        center_jittering=0.0,
        block_rot=False,
        max_rot=0.0 * 3.1415926,
        hue=0.15,
        saturation=0.5,
        contrast=0.5,
        brightness=0.5,
        blur_radius=0.5,
        query=None,
        sides="right",
        assets_root=assets_root),
)

keys = [
    'BaseQueries.JOINT_VIS',
    'BaseQueries.IMAGE',
    'BaseQueries.CAM_INTR',
    'TransQueries.CAM_INTR',
    'TransQueries.AFFINETRANS',
    'BaseQueries.OBJ_VERTS_2D',
    'TransQueries.OBJ_VERTS_2D',
    'BaseQueries.OBJ_VIS_2D',
    'BaseQueries.JOINTS_2D',
    'TransQueries.JOINTS_2D',
    'BaseQueries.HAND_VERTS_2D',
    'TransQueries.HAND_VERTS_2D',
    'BaseQueries.JOINTS_3D',
    'BaseQueries.CENTER_3D',
    'TransQueries.JOINTS_3D',
    'TransQueries.CENTER_3D',
    'BaseQueries.HAND_VERTS_3D',
    'TransQueries.HAND_VERTS_3D',
    'BaseQueries.OBJ_VERTS_3D',
    'TransQueries.OBJ_VERTS_3D',
    'BaseQueries.OBJ_CAN_VERTS',
    'BaseQueries.OBJ_CAN_SCALE',
    'BaseQueries.OBJ_CAN_TRANS',
    'TransQueries.IMAGE',
    'BaseQueries.OBJ_TRANSF',
    'TransQueries.OBJ_TRANSF',
    'BaseQueries.HAND_POSE_WRT_CAM',
]


meta_keys = [
    'BaseQueries.IMAGE_PATH',
    'BaseQueries.SIDE',
    'MetaQueries.SAMPLE_IDENTIFIER'
]



train_pipeline = [dict(type='Collect3D', keys=keys, meta_keys=[],)]
test_pipeline = [
    dict(type='ToTensor', keys=keys),
    dict(type='Collect3D', keys=keys, meta_keys=[],)
]

data = dict(
    samples_per_gpu=2, # bz = 320
    workers_per_gpu=0,
    train=dict(type='HODataset', **data_cfg, pipeline=train_pipeline),
    val=dict(type='HODataset', **data_cfg, pipeline=test_pipeline),
    test=dict(type='HODataset', **data_cfg, pipeline=test_pipeline))


# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[])
runner = dict(type='EpochBasedRunner', max_epochs=300)


checkpoint_config = dict(interval=100)
evaluation = dict(interval=100)


log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = False
