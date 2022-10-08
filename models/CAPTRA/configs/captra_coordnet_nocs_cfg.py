obj_category = '1'
dataset_type = 'SingleFrameDataset'

obj_cfg = {'1': {'name': 'bottle',
   'sym': True,
   'type': 'revolute',
   'num_parts': 1,
   'num_joints': 0,
   'tree': [-1],
   'parts_map': [[0]],
   'augment_idx': {'x': [-1], 'y': [-1], 'z': [-1]},
   'main_axis': [],
   'template': '2a9817a43c5b3983bb13793251b29587',
   'bad_ins': []},
  '2': {'name': 'bowl',
   'sym': True,
   'type': 'revolute',
   'num_parts': 1,
   'num_joints': 0,
   'tree': [-1],
   'parts_map': [[0]],
   'augment_idx': {'x': [-1], 'y': [-1], 'z': [-1]},
   'main_axis': [],
   'template': '1b4d7803a3298f8477bdcb8816a3fac9',
   'bad_ins': []},
  '3': {'name': 'camera',
   'sym': False,
   'type': 'revolute',
   'num_parts': 1,
   'num_joints': 0,
   'tree': [-1],
   'parts_map': [[0]],
   'augment_idx': {'x': [-1], 'y': [-1], 'z': [-1]},
   'main_axis': [],
   'template': '5d42d432ec71bfa1d5004b533b242ce6',
   'bad_ins': ['1298634053ad50d36d07c55cf995503e',
    '2153bc743019671ae60635d9e388f801',
    '22217d5660444eeeca93934e5f39869',
    '290abe056b205c08240c46d333a693f',
    '39419e462f08dcbdc98cccf0d0f53d7',
    '4700873107186c6e2203f435e9e6785',
    '550aea46c75351a387cfe978d99ba05d',
    '60923e8a6c785a8755a834a7aafb0236',
    '6ed69b00b4632b6e07718ee10b83e10',
    '7077395b60bf4aeb3cb44973ec1ffcf8',
    '87b8cec4d55b5f2d75d556067d060edf',
    '97cd28c085e3754f22c69c86438afd28',
    'a9408583f2c4d6acad8a06dbee1d115',
    'b27815a2bde54ad3ab3dfa44f5fab01',
    'b42c73b391e14cb16f05a1f780f1cef',
    'c3e6564fe7c8157ecedd967f62b864ab',
    'c802792f388650428341191174307890',
    'd680d61f934eaa163b211460f022e3d9',
    'd9bb9c5a48c3afbfb84553e864d84802',
    'e3dc17dbde3087491a722bdf095986a4',
    'e57aa404a000df88d5d4532c6bb4bd2b',
    'eb86c8c2a20066d0fb1468f5fc754e02',
    'ee58b922bd93d01be4f112f1b3124b84',
    'fe669947912103aede650492e45fb14f',
    'ff74c4d2e710df3401a67448bf8fe08']},
  '4': {'name': 'can',
   'sym': True,
   'type': 'revolute',
   'num_parts': 1,
   'num_joints': 0,
   'tree': [-1],
   'parts_map': [[0]],
   'augment_idx': {'x': [-1], 'y': [-1], 'z': [-1]},
   'main_axis': [],
   'template': '97ca02ee1e7b8efb6193d9e76bb15876',
   'bad_ins': []},
  '5': {'name': 'laptop',
   'sym': False,
   'type': 'revolute',
   'num_parts': 1,
   'num_joints': 0,
   'tree': [-1],
   'parts_map': [[0]],
   'augment_idx': {'x': [-1], 'y': [-1], 'z': [-1]},
   'main_axis': [],
   'template': '3e9af28eb2d6e216a4e3429ccb8eaf16',
   'bad_ins': []},
  '6': {'name': 'mug',
   'sym': False,
   'type': 'revolute',
   'num_parts': 1,
   'num_joints': 0,
   'tree': [-1],
   'parts_map': [[0]],
   'augment_idx': {'x': [-1], 'y': [-1], 'z': [-1]},
   'main_axis': [],
   'template': 'a6d9f9ae39728831808951ff5fb582ac',
   'bad_ins': []}}


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
           basepath='/disk1/data/captar',
           num_points=4096,
           dataset_length=None,
           synthetic=True,
           nocs_data=True,
           num_expr='1_bottle_coord_new',
           data_radius=0.6,
           pose_perturb={'type': 'normal', 's': 0.02, 'r': 5.0, 't': 0.03}
           )


pipeline = [dict(type='Collect3D', keys=['labels', 'points', 'nocs',
                                         'crop_pose_scale', 'crop_pose_translation',
                                         'nocs2camera_scale', 'nocs2camera_translation', 'nocs2camera_translation', 'nocs2camera_rotation',
                                         'nocs_corners', 'points_mean'],
                 meta_keys=['ori_path', 'path'])]

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