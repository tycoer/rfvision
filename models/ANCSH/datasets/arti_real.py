import os
import sys

import numpy as np
import open3d as o3d
import mmcv
import copy
import pickle
import json

from torch.utils.data import Dataset

from mmdet3d.datasets.pipelines import Compose
from mmdet3d.datasets.builder import DATASETS



@DATASETS.register_module()
class ArtiRealDataset(Dataset):
    CLASSES = None
    def __init__(self, ann_file,
                 pipeline,
                 img_prefix,
                 intrinsics_path,
                 test_mode=False,
                 domain='real',
                 n_parts=3,
                 is_gen=False,
                 **kwargs):
        self.is_gen = is_gen
        self.n_parts = n_parts
        self.img_prefix = img_prefix
        self.domain = domain
        self.test_mode = test_mode
        self.camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsics_path)
        self.annotation_path = os.path.join(self.img_prefix, 'annotations')
        self.sample_list = mmcv.list_from_file(ann_file)

        self.norm_factors, self.corner_pts = self.fetch_factors_nocs()
        self.all_joint_ins = self.fetch_joints_params()

        if not self.test_mode:
            self._set_group_flag()

        self.pipeline = Compose(pipeline)

    def pre_pipeline(self, results):
        results['camera_intrinsic'] = self.camera_intrinsic
        results['img_prefix'] = self.img_prefix
        results['domain'] = self.domain

        scene, video, h5_file = results['sample_name'].split('/')
        filename, instance_id = h5_file.split('.h5')[0].split('_')
        instance_id = int(instance_id)
        data_info = json.load(open(os.path.join(self.annotation_path, scene, video, filename + '.json')))
        instance_info = data_info['instances'][instance_id]

        img_width = data_info['width']
        img_height = data_info['height']
        bbox = instance_info['bbox']

        urdf_id = instance_info['urdf_id']
        joint_ins = self.all_joint_ins[urdf_id]
        category_id = instance_info['category_id']
        norm_factors = self.norm_factors[urdf_id]
        corner_pts = self.corner_pts[urdf_id]

        results.update(dict(instance_info=instance_info,
                            img_width=img_width,
                            img_height=img_height,
                            bbox=bbox,
                            category_id=category_id,
                            joint_ins=joint_ins,
                            norm_factors=norm_factors,
                            corner_pts=corner_pts))

    def prepare_train_sample(self, idx):
        sample_name = self.sample_list[idx]
        results = dict(sample_name=sample_name)
        # ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_sample(self, idx):
        sample_name = self.sample_list[idx]
        results = dict(sample_name=sample_name)
        # img_info = self.img_infos[idx]
        # results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_sample(idx)
        else:
            return self.prepare_train_sample(idx)

    def __len__(self):
        return len(self.sample_list)

    def _set_group_flag(self):
        """Set flag to 1
        """
        self.flag = np.ones(len(self), dtype=np.uint8)

    def fetch_factors_nocs(self):
        norm_factors = {}
        corner_pts = {}
        urdf_metas = json.load(open(self.img_prefix + '/urdf_metas.json'))['urdf_metas']
        for urdf_meta in urdf_metas:
            norm_factors[urdf_meta['id']] = np.array(urdf_meta['norm_factors'])
            corner_pts[urdf_meta['id']] = np.array(urdf_meta['corner_pts'])

        return norm_factors, corner_pts

    def fetch_joints_params(self):
        joint_ins = {}
        urdf_metas = json.load(open(self.img_prefix + '/urdf_metas.json'))['urdf_metas']
        for urdf_meta in urdf_metas:
            if urdf_meta == []:
                continue
            joint_ins[urdf_meta['id']] = dict(xyz=[], axis=[], type=[], parent=[], child=[])

            for n in range(self.n_parts - 1):
                if n == 0:
                    joint_ins[urdf_meta['id']]['xyz'].append([0., 0., 0.])
                    joint_ins[urdf_meta['id']]['axis'].append([0., 0., 0.])
                    joint_ins[urdf_meta['id']]['type'].append(None)
                    joint_ins[urdf_meta['id']]['parent'].append(None)
                    joint_ins[urdf_meta['id']]['child'].append(None)
                    continue
                x, y, z = urdf_meta['joint_xyz'][n-1][::-1]
                joint_ins[urdf_meta['id']]['xyz'].append([y, x, z])
                r, p, y = urdf_meta['joint_rpy'][n - 1][::-1]
                joint_ins[urdf_meta['id']]['axis'].append([p, -r, y])
                joint_ins[urdf_meta['id']]['type'].append(urdf_meta['joint_types'][n-1])
                joint_ins[urdf_meta['id']]['parent'].append(urdf_meta['joint_parents'][n-1])
                joint_ins[urdf_meta['id']]['child'].append(urdf_meta['joint_children'][n-1])

        return joint_ins


if __name__ == '__main__':
    from mmdet.datasets.builder import build_dataloader
    img_norm_cfg = dict(
        mean=[123.7, 116.8, 103.9], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
        dict(type='LoadArtiPointData'),
        dict(type='LoadArtiNOCSData'),
        dict(type='LoadArtiJointData'),
        dict(type='CreateArtiJointGT'),
        dict(type='DownSampleArti', num_points=1024),
        # dict(type='LoadAnnotationsNOCS', with_mask=True, with_coord=True),
        # dict(type='ResizeNOCS', min_dim=480, max_dim=640, padding=True),
        # dict(type='ExtractBBoxFromMask'),
        # dict(type='RandomFlip', flip_ratio=0.5),
        # dict(type='Normalize', **img_norm_cfg, is_nocs=True),
        # dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundleArti'),
        dict(type='Collect', keys=['parts_pts', 'parts_pts_feature', 'parts_cls', 'mask_array',
                                   'nocs_p', 'nocs_g', 'offset_heatmap',
                                   'offset_unitvec', 'joint_orient', 'joint_cls',
                                   'joint_cls_mask', 'joint_params'],
             meta_keys=['img_prefix', 'sample_name', 'norm_factors', 'corner_pts',
                        'joint_ins']),
    ]

    dataset = ArtiRealDataset(ann_file='data/real_data/stapler/train.txt',
                              pipeline=train_pipeline,
                              img_prefix='data/real_data/stapler',
                              intrinsics_path='data/real_data/stapler/camera_intrinsic.json')
    print(len(dataset))

    #for i in range(len(dataset)):
        #res = dataset[i]
        #for k in ['color_img', 'depth_img', 'gt_labels', 'gt_bboxes', 'gt_masks', 'gt_coords', 'scales']:
            #print(res[k].data.shape)
    dataloader = build_dataloader(dataset, 2, 2)

    for i, da in enumerate(dataloader):
        print(i, da)
