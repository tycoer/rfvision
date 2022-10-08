from models.CAPTRA.datasets.captra_dataset import SequenceData
from models.CAPTRA.test.test_rotnet import obj_cfg

obj_category = '1'


data_cfg = dict(obj_cfg=obj_cfg[obj_category],
           obj_category=obj_category,
           basepath='/disk1/data/captar',
           num_points=4096,
           dataset_length=None,
           synthetic=True,
           nocs_data=True,
           num_expr='1_bottle_coord_new',
           data_radius=0.6,
           pose_perturb={'type': 'normal', 's': 0.02, 'r': 5.0, 't': 0.03},
           num_frames=100,
           )

if __name__ == '__main__':
    dataset = SequenceData(cfg=data_cfg, mode='real_test')
    data = dataset[0]