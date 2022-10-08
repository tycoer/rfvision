assets_root="/home/hanyang/rfvision/models/CPF/assets"
data_root="/ssdisk1/cpf/data"
datasets = ['fhb']

model=dict(
    type='CPFPicrOnline',
    backbone=dict(
        type='PicrHourglassPointNet',
        hg_stacks=2,
        hg_blocks=1,
        hg_classes=64,
        obj_scale_factor=0.0001,
        honet_resnet_version=18,
        honet_center_idx=9,
        honet_mano_lambda_recov_joints3d=0.5,
        honet_mano_lambda_recov_verts3d=0,
        honet_mano_lambda_shape=5e-07,
        honet_mano_lambda_pose_reg=5e-06,
        honet_obj_lambda_recov_verts3d=0.5,
        honet_obj_trans_factor=100,
        honet_mano_fhb_hand=False,
        mean_offset=0.010,
        std_offset=0.005,
        maximal_angle=3.1415926 / 24,
        init_honet_ckpt=None, # tycoer
        init_picr_ckpt=None, # tycoer
        assets_root=assets_root
    ),

    loss_contact=dict(
        type='VertexContactLoss',
        contact_lambda_vertex_contact=10,
        contact_lambda_contact_region=10,
        contact_lambda_anchor_elasti=1,
        focal_loss_alpha=0.9,
        focal_loss_gamma=2,
        region_focal_loss_alpha='fhb',),
)


data_cfg = dict(
    datasets='fhb',
    data_root=data_root,
    data_split='train',
    split_mode='actions',
    use_cache=False,
    mini_factor=1,
    center_idx=9,
    enable_contact=True,
    filter_no_contact=True,
    filter_thresh=10.0,  # useless
    # query=query,
    like_v1=True,
    block_rot=True,
    max_rot=True,
    scale_jittering=0,
    center_jittering=0.1,
    synt_factor=(1, ),
    assets_root=assets_root,
    num_points=3000,

    init_cfgs=dict(
        data_root=data_root,
        data_split="train",
        njoints=21,
        use_cache=True,
        filter_no_contact=True,
        filter_thresh=10.0,
        mini_factor=1.0,
        center_idx=9,
        scale_jittering=0.0,
        center_jittering=0.1,
        block_rot=False,
        max_rot= 3.1415926 / 12,
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

    'ContactQueries.VERTEX_CONTACT',
    'ContactQueries.CONTACT_REGION_ID',
    'ContactQueries.CONTACT_ANCHOR_ID',
    'ContactQueries.CONTACT_ANCHOR_DIST',
    'ContactQueries.CONTACT_ANCHOR_ELASTI',
    'ContactQueries.CONTACT_ANCHOR_PADDING_MASK',

    'CollateQueries.PADDING_MASK'
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
    samples_per_gpu=2, # bz = 8
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
# cudnn_benchmark = True

