specs = {
        # model para
        "CodeLength" : 253,
        "Articulation" : True,
        "NumAtcParts" : 2,
        "TrainWithParts" : False,
        "ClampingDistance" : 0.1,
        # dataset para
        "TrainSplit": "examples/splits/sm_refrigerator_6_angle_train.json",
        "TestSplit": "examples/splits/sm_refrigerator_6_angle_test.json",
        # meta
        'Class': 'refrigerator'
      }



model=dict(
    type='ASDF',
    latent_size=specs['CodeLength'],
    articulation=specs["Articulation"],  # for development
    num_atc_parts=specs["NumAtcParts"],
    do_sup_with_part=specs["TrainWithParts"],

    dims=[512, 512, 512, 768, 512, 512, 512, 512],
    dropout=[0, 1, 2, 3, 4, 5, 6, 7],
    dropout_prob=0.2,
    norm_layers=[0, 1, 2, 3, 4, 5, 6, 7],
    latent_in=[4],
    init_cfg=dict(type='Pretrained', checkpoint='models/ASDF/examples/checkpoints/laptop/ModelParameters/1000.pth') # test
    # init_cfg=None,
)

data_cfg=dict(
    data_source='./data',
    split='models/ASDF/examples/splits/sm_door_6_angle_train.json',
    subsample=16000,
    load_ram=False,
    articulation=specs["Articulation"],
    num_atc_parts=specs["NumAtcParts"],
)

pipeline = [dict(type='Collect3D', keys=['indices', 'sdf_data', 'atc', 'instance_idx'], meta_keys=[])]
data = dict(
    samples_per_gpu=32, # bz 32
    workers_per_gpu=0,
    train=dict(type='ASDFDataset', pipeline=pipeline, **data_cfg),
    val=dict(type='ASDFDataset', pipeline=pipeline, **data_cfg),
    test=dict(type='ASDFDataset', pipeline=pipeline, **data_cfg),
)

optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[])
runner = dict(type='EpochBasedRunner', max_epochs=1000)
checkpoint_config=dict(interval=100)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluation = dict(interval=100)
# runtime settings
total_epochs = 1000
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
