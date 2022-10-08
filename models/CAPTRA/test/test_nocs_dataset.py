from models.CAPTRA.datasets.nocs_dataset import NOCSDataset

bottle_info ={
    'name': 'bottle',
    'sym': True,
    'type': 'revolute',
    'num_parts': 1,
    'num_joints': 0,
    'tree': [-1],
    'parts_map': [[0]],
    'augment_idx': {'x': [-1], 'y': [-1], 'z': [-1]},
    'main_axis': [],
    'template': '2a9817a43c5b3983bb13793251b29587',
    'bad_ins': []
}


perturb_cfg={'type': 'normal', 's': 0.02, 'r': 5.0, 't': 0.03}
if __name__ == '__main__':
    cfg=dict(mode='train',
             root_dset='/disk1/data/captar',
             obj_category='1',
             obj_info=bottle_info,
             num_expr='1_bottle_rot_new',
             num_points=4096,
             truncate_length=None,
             radius=0.6,
             perturb_cfg=perturb_cfg,
             downsampling=None
             )

    dataset = NOCSDataset(**cfg)
    data = dataset[0]