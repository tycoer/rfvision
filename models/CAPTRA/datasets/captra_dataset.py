import os
import sys
import argparse
import random
from copy import deepcopy
from tqdm import tqdm
from abc import abstractmethod

from os.path import join as pjoin
from torch.utils.data import Dataset, DataLoader
from .sapien_dataset import SAPIENDataset
from .sapien_real_dataset import SAPIENRealDataset
from .bmvc_dataset import BMVCDataset
from .nocs_dataset import NOCSDataset
from ..utils.data_utils import get_seq_file_list_index
from ..utils.data_transforms import shuffle, subtract_mean, add_corners
from ..utils.io import ensure_dirs

from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.pipelines import Compose
import numpy as np

class PointData(Dataset):
    def __init__(self, cfg, mode='train', downsampling=None):
        super(PointData, self).__init__()
        self.cfg = cfg
        self.mode = mode
        obj_cfg = cfg['obj_cfg']
        root_dir = cfg['basepath']
        ctgy = cfg['obj_category']
        obj_info = obj_cfg
        # num_expr = cfg['num_expr']
        num_expr = f"1_{obj_cfg['name']}_coord_new" if 'name' in obj_cfg else f'{ctgy}_coord_new'
        num_points = cfg['num_points']
        truncate_length = cfg['dataset_length']
        synthetic = cfg['synthetic']
        perturb = 'perturb' in obj_cfg and obj_cfg['perturb']
        preproc_dir = pjoin(root_dir, 'preproc', ctgy)
        ensure_dirs(preproc_dir)
        self.synthetic = synthetic
        self.nocs_data = 'nocs_data' in cfg and cfg['nocs_data']
        self.real_data = self.mode in ['real_test'] and not self.nocs_data
        self.bmvc_data = 'bmvc' in self.mode
        if self.nocs_data:
            self.dataset = NOCSDataset(root_dset=root_dir, obj_category=ctgy, obj_info=obj_info,
                                       num_expr=num_expr, num_points=num_points,
                                       mode=mode, truncate_length=truncate_length, synthetic=synthetic,
                                       radius=cfg['data_radius'], perturb_cfg=cfg['pose_perturb'],
                                       device=None, downsampling=downsampling)
        elif self.real_data:
            self.dataset = SAPIENRealDataset(root_dset=root_dir, obj_category=ctgy, obj_info=obj_info,
                                             num_expr=num_expr, num_points=num_points,
                                             truncate_length=truncate_length)
        elif self.bmvc_data:
            self.dataset = BMVCDataset(root_dset=root_dir, obj_category=ctgy, track=int(self.mode.split('_')[-1]),
                                       truncate_length=truncate_length)
        else:
            self.dataset = SAPIENDataset(root_dset=root_dir, obj_category=ctgy, obj_info=obj_info,
                                         num_expr=num_expr, num_points=num_points,
                                         mode=mode, truncate_length=truncate_length,
                                         synthetic=synthetic, perturb=perturb, device=None)
        self.ins_info = self.dataset.ins_info
        self.data = {}
        self.num_points = cfg['num_points']
        self.tree = obj_cfg['tree']
        self.root = [p for p in range(len(self.tree)) if self.tree[p] == -1][0]
        self.num_parts = len(self.tree)

    def retrieve_single_frame(self, item):
        data = self.dataset[item]
        data_dict = data['data']
        meta_dict = data['meta']

        def reshape(x):  # from [N, x, x] to [C, N]
            x = x.reshape(x.shape[0], -1)
            x = x.swapaxes(0, 1)
            return x

        for key in data_dict.keys():
            if key in ['labels']:
                continue
            data_dict[key] = reshape(data_dict[key])
        data_dict['meta'] = meta_dict

        item_idx = data['meta']['path'].split('.')[-2].split('/')[-3]
        if 'nocs_corners' not in data_dict['meta'] and not self.real_data:
            data_dict = add_corners(data_dict, self.ins_info[item_idx])

        return data_dict

    @abstractmethod
    def retrieve_item(self, index):  # if not present, retrieve
        pass

    def save_all_items(self):
        for i in tqdm(range(len(self))):
            self.retrieve_item(i)

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

class SingleFrameData(PointData):
    def __init__(self, cfg, mode='train', downsampling=None):
        super(SingleFrameData, self).__init__(cfg, mode, downsampling)
        self.len = len(self.dataset)
        self.index_dict = {}
        self.cfg = cfg

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ret = None
        if idx not in self.index_dict:
            self.index_dict[idx] = idx
        final_idx = self.index_dict[idx]
        while ret is None:
            ret = deepcopy(self.retrieve_single_frame(final_idx))
            if ret is None:
                final_idx = (final_idx + random.randint(1, self.len - 1)) % self.len
                self.index_dict[idx] = final_idx
        ret = shuffle(ret)
        ret = subtract_mean(ret)
        return ret


class SequenceData(PointData):
    def __init__(self, cfg, mode='train', downsampling=None):
        super(SequenceData, self).__init__(cfg, mode, downsampling)
        if 'real' not in self.mode and 'bmvc' not in self.mode:
            self.num_frames = cfg['obj']['num_frames']
            print("dataset length", len(self.dataset))
            if len(self.dataset) >= self.num_frames:
                assert len(self.dataset) % self.num_frames == 0, "Total #frames mismatch with #frames/video"
            else:
                self.num_frames = len(self.dataset)
            self.len = len(self.dataset) // self.num_frames
            self.index_dict = {}
        elif 'bmvc' in self.mode:
            self.len = 1
        else:
            self.seq_start = get_seq_file_list_index(self.dataset.file_list)
            print('seq start', self.seq_start)
            self.len = len(self.seq_start) - 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.bmvc_data:
            seq_data = []
            for i in range(len(self.dataset)):
                data = deepcopy(self.retrieve_single_frame(i))
                data = shuffle(data)
                seq_data.append(data)
        elif self.nocs_data or self.real_data:
            seq_data = []
            for i in range(self.seq_start[idx], self.seq_start[idx + 1]):
                data = deepcopy(self.retrieve_single_frame(i))
                data = shuffle(data)
                seq_data.append(data)
        else:
            if idx not in self.index_dict:
                self.index_dict[idx] = idx
            final_idx = self.index_dict[idx]

            while True:
                cur_flag = True
                seq_data = []
                for i in range(final_idx * self.num_frames, (final_idx + 1) * self.num_frames):
                    data = deepcopy(self.retrieve_single_frame(i))
                    if data is None:
                        cur_flag = False
                        break
                    data = shuffle(data)
                    seq_data.append(data)
                if not cur_flag:
                    final_idx = (final_idx + random.randint(1, self.len - 1)) % self.len
                    self.index_dict[idx] = final_idx
                else:
                    break
        ret = []
        for data in seq_data:
            data = subtract_mean(data)
            ret.append(data)
        return ret

@DATASETS.register_module()
class SingleFrameDataset(SingleFrameData):
    CLASSES = None
    def __init__(self,
                 cfg,
                 mode='train',
                 pipeline=None,
                 ):
        super(SingleFrameDataset, self).__init__(cfg=cfg, mode=mode)
        self.pipeline = pipeline
        if self.pipeline is not None:
            self.pipeline = Compose(self.pipeline)
        self.flag = np.zeros(len(self), dtype='int64')
    def __getitem__(self, item):
        results = super(SingleFrameDataset, self).__getitem__(item)
        results = self.processing(results)
        if self.pipeline is not None:
            results = self.pipeline(results)
        return results

    def processing(self, results):
        if self.nocs_data:
            results['crop_pose_scale'], results['crop_pose_translation'] = results['meta']['crop_pose'][0]['scale'], results['meta']['crop_pose'][0]['translation']
            results['ori_path'] = results['meta']['ori_path']
            results['nocs2camera_scale'], results['nocs2camera_translation'], results['nocs2camera_rotation'] = (results['meta']['nocs2camera'][0]['scale'],
                                                                                                                 results['meta']['nocs2camera'][0]['translation'],
                                                                                                                 results['meta']['nocs2camera'][0]['rotation'],)
        else:
            scale, rotation, translation = np.zeros((self.num_parts)), np.zeros((self.num_parts, 3, 3)), np.zeros((self.num_parts, 3, 1))
            for i, info in enumerate(results['meta']['nocs2camera']):
                scale[i], rotation[i], translation[i] = info['scale'], info['rotation'], info['translation']



            results['nocs2camera_scale'], results['nocs2camera_translation'], results['nocs2camera_rotation'] = scale, translation, rotation
        results['nocs_corners'] = results['meta']['nocs_corners']
        results['path'] = results['meta']['path']
        results['points_mean'] = results['meta']['points_mean']
        return results

@DATASETS.register_module()
class SequenceDataset(SequenceData):
    CLASSES = None
    def __init__(self,
                 cfg,
                 mode='train',
                 pipeline=None,
                 ):
        super(SequenceDataset, self).__init__(cfg=cfg, mode=mode)
        self.pipeline = pipeline
        if self.pipeline is not None:
            self.pipeline = Compose(self.pipeline)
        self.flag = np.zeros(len(self), dtype='int64')
    def __getitem__(self, item):
        results = super(SequenceDataset, self).__getitem__(item)
        if self.pipeline is not None:
            results = self.pipeline(results)
        return results


def get_dataloader(cfg, mode='train', shuffle=None, downsampling=None):
    if shuffle is None:
        shuffle = (mode == 'train')
    if 'track' in cfg['network']['type']:
        dataset = SequenceData(cfg, mode=mode, downsampling=downsampling)
    else:
        dataset = SingleFrameData(cfg, mode=mode, downsampling=downsampling)
    batch_size = cfg['batch_size']
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cfg['num_workers'])


def parse_args():
    parser = argparse.ArgumentParser('Dataset')
    parser.add_argument('--config', type=str, default='config.yml', help='path to config.yml')
    parser.add_argument('--obj_config', type=str, default=None)
    parser.add_argument('--obj_category', type=str, default=None)
    parser.add_argument('--experiment_dir', type=str, default=None, help='root dir for all outputs')
    parser.add_argument('--num_points', type=int,  default=None, help='Point Number [default: 1024]')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch Size during training [default: 16]')
    parser.add_argument('--worker', type=int, default=None, help='Batch Size during training [default: 16]')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    cfg = get_config(args, save=False)

