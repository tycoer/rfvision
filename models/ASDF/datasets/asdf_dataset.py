#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import re
from ..utils import workspace as ws
import json
#tycoer
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.pipelines import Compose

class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        articulation=False,
        num_atc_parts=1,
        **kwargs
    ):
        # tycoer
        with open(split, "r") as f:
            split = json.load(f)

        self.subsample = subsample
        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)
        self.articualtion = articulation
        self.num_atc_parts = num_atc_parts

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

                if self.articualtion==True:
                    if self.num_atc_parts==1:
                        atc = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
                        instance_idx = int(re.split('/', filename)[-1][:4])
                        self.loaded_data.append(
                            (
                            [
                                pos_tensor[torch.randperm(pos_tensor.shape[0])],
                                neg_tensor[torch.randperm(neg_tensor.shape[0])],
                            ],
                            atc,
                            instance_idx,
                            )
                        )
                    if self.num_atc_parts==2:
                        atc1 = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
                        atc2 = torch.Tensor([float(re.split('/', filename)[-1][11:15])])
                        instance_idx = int(re.split('/', filename)[-1][:4])
                        self.loaded_data.append(
                            (
                            [
                                pos_tensor[torch.randperm(pos_tensor.shape[0])],
                                neg_tensor[torch.randperm(neg_tensor.shape[0])],
                            ],
                            [atc1, atc2],
                            instance_idx,
                            )
                        )

                else:
                    self.loaded_data.append(
                        [
                            pos_tensor[torch.randperm(pos_tensor.shape[0])],
                            neg_tensor[torch.randperm(neg_tensor.shape[0])],
                        ],
                    )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample, self.articualtion, self.num_atc_parts),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample, self.articualtion, self.num_atc_parts), idx

def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename, articulation=False, num_atc_parts=1):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])
    if articulation==True:
        if num_atc_parts==1:
            atc = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
            instance_idx = int(re.split('/', filename)[-1][:4])
            return ([pos_tensor, neg_tensor], atc, instance_idx)
        if num_atc_parts==2:
            atc1 = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
            atc2 = torch.Tensor([float(re.split('/', filename)[-1][11:15])])
            instance_idx = int(re.split('/', filename)[-1][:4])
            return ([pos_tensor, neg_tensor], torch.Tensor([atc1, atc2]), instance_idx)
    else:
        return [pos_tensor, neg_tensor]

def read_sdf_samples_into_ram_rbo(filename, articulation=False, num_atc_parts=1):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])
    if articulation==True:
        if num_atc_parts==1:
            atc = torch.Tensor([float(re.split('/', filename)[-1][-8:-4])])
            instance_idx = int(re.split('/', filename)[-1][:4])
            return ([pos_tensor, neg_tensor], atc, instance_idx)
    else:
        return [pos_tensor, neg_tensor]

def unpack_sdf_samples(filename, subsample=None, articulation=False, num_atc_parts=1):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
    
    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    if articulation==True:
        if num_atc_parts==1:
            atc = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
            instance_idx = int(re.split('/', filename)[-1][:4])
            return (samples, atc, instance_idx)
        if num_atc_parts==2:
            atc1 = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
            atc2 = torch.Tensor([float(re.split('/', filename)[-1][11:15])])
            instance_idx = int(re.split('/', filename)[-1][:4])
            return (samples, torch.Tensor([atc1, atc2]), instance_idx)
    else:
        return samples


def unpack_sdf_samples_from_ram(data, subsample=None, articulation=False, num_atc_parts=1):
    if subsample is None:
        return data
    if articulation==True:
        pos_tensor = data[0][0]
        neg_tensor = data[0][1]
        atc = data[1]
        instance_idx = data[2]
    else:
        pos_tensor = data[0]
        neg_tensor = data[1]        

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    #pos_start_ind = random.randint(0, pos_size - half)
    #sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if pos_size <= half:
        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    else:
        pos_start_ind = random.randint(0, pos_size - half)
        sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    if articulation==True:
        return (samples, atc, instance_idx)
    else:
        return samples

@DATASETS.register_module()
class ASDFDataset(SDFSamples):
    CLASSES=None
    def __init__(self,
                 data_source,
                 split,
                 subsample,
                 load_ram=False,
                 articulation=False,
                 num_atc_parts=1,
                 pipeline=None,
                 **kwargs
                 ):
        super(ASDFDataset, self).__init__(
            data_source=data_source,
            split=split,
            subsample=subsample,
            load_ram=load_ram,
            articulation=articulation,
            num_atc_parts=num_atc_parts
        )

        self.pipeline = pipeline
        if self.pipeline is not None:
            self.pipeline = Compose(self.pipeline)
        self.flag = np.zeros(len(self), dtype='int32')

    def __getitem__(self, item):
        all_sdf_data, indices = super().__getitem__(item)
        results = dict(indices=indices,
                       sdf_data=all_sdf_data[0],
                       atc=all_sdf_data[1],
                       instance_idx=all_sdf_data[2],
                       )

        if self.pipeline is not None:
            results = self.pipeline(results)
        return results
