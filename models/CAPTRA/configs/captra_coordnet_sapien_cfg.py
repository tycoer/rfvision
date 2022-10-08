obj_category = 'drawers'
dataset_type = 'SingleFrameDataset'

obj_cfg =  {'glasses': {'sym': False,
  'type': 'revolute',
  'num_parts': 3,
  'num_joints': 2,
  'tree': [2, 2, -1],
  'exemplar': '101300',
  'parts_map': [[0], [1], [2]],
  'test_list': ['101839',
   '101848',
   '101860',
   '101326',
   '101868',
   '102591',
   '102596',
   '103028'],
  'train_list': None,
  'augment_idx': {'x': [-1, -1, -1], 'y': [-1, 0, 0], 'z': [-1, 0, -1]},
  'main_axis': [1, 1]},
 'scissors': {'sym': False,
  'type': 'revolute',
  'num_parts': 2,
  'num_joints': 1,
  'tree': [-1, 0],
  'parts_map': [[0], [1]],
  'test_list': ['10559', '10564', '11029'],
  'train_list': None,
  'augment_idx': {'x': [-1, 0], 'y': [-1, 0], 'z': [-1, 0]},
  'main_axis': [1]},
 'drawers': {'sym': False,
  'type': 'prismatic',
  'num_parts': 4,
  'num_joints': 3,
  'tree': [3, 3, 3, -1],
  'parts_map': [[0], [1], [2], [3]],
  'test_list': ['46440', '46123'],
  'train_list': None,
  'augment_idx': {'x': [-1, 0, 0, 0], 'y': [-1, 0, 0, 0], 'z': [-1, 0, 0, 0]},
  'main_axis': [2, 2, 2]},
 'laptop': {'sym': False,
  'type': 'revolute',
  'num_parts': 2,
  'num_joints': 1,
  'tree': [-1, 0],
  'parts_map': [[0], [1]],
  'test_list': ['10101', '10270', '10356', '11156', '11405', '11581'],
  'train_list': None,
  'augment_idx': {'x': [-1, 0], 'y': [-1, -1], 'z': [-1, 0]},
  'main_axis': [0]}}


model=dict(
    type='CaptraCoordNet',
    obj_cfg=obj_cfg[obj_category],
    backbone_out_dim=128,
    extra_dims=1,
    nocs_head_dims=[128],
    pwm_num = 128,
    pose_perturb_cfg=dict(type='normal',
                          scale=0.02,
                          translation=0.03,
                          rotation=5.0),
    loss_t_weight=5.0,
    loss_s_weight=5.0,
    loss_corner_weight=10,
    loss_nocs_weight=10,
    loss_seg_weight=1.0,
    loss_nocs_dist_weight=5.0,
    loss_nocs_pwm_weight=5.0,
    init_cfg=None)

data_cfg = dict(obj_cfg=obj_cfg[obj_category],
           obj_category=obj_category,
           basepath='/disk1/data/captar/sapien_data',
           num_points=4096,
           dataset_length=None,
           synthetic=True,
           num_expr='drawers_rot_new',
           data_radius=0.6,
           pose_perturb={'type': 'normal', 's': 0.02, 'r': 5.0, 't': 0.03}
           )


pipeline = [dict(type='Collect3D', keys=['labels', 'points', 'nocs',
                                         'nocs2camera_scale', 'nocs2camera_translation', 'nocs2camera_translation', 'nocs2camera_rotation',
                                         'nocs_corners', 'points_mean'],
                 meta_keys=['path'])]

data = dict(
    samples_per_gpu=16, # bz = 12
    workers_per_gpu=0,
    train=dict(type=dataset_type, cfg=data_cfg, mode='train', pipeline=pipeline),
    val=dict(type=dataset_type, cfg=data_cfg, mode='real_test', pipeline=pipeline),
    test=dict(type=dataset_type, cfg=data_cfg, mode='real_test', pipeline=pipeline))


optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[250, 750])
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