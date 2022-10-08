import os
from os.path import join
from models.CAPTRA.scripts.checkpoint_offical2rf_sapien import main
from models.CAPTRA.configs.obj_cfgs_sapien import sapien_obj_cfgs
from models.CAPTRA.models.captra_coordnet import CoordNet
from models.CAPTRA.models.captra_rotnet import PartCanonNet
import argparse

def get_sapien_keys():
    rf_sapien_coordnet_keys = dict()
    rf_sapien_rotnet_keys = dict()
    for obj_category in sapien_obj_cfgs.keys():
        rotnet = PartCanonNet(
            obj_cfg=sapien_obj_cfgs[obj_category],
            backbone_out_dim=128,
            network_type='rot',

        )
        coordnet = CoordNet(
            nocs_head_dims=[128],
            obj_cfg=sapien_obj_cfgs[obj_category],
            backbone_out_dim=128,
            extra_dims=1,
        )
        rf_sapien_coordnet_keys[obj_category] = [f'coordnet.{k}'for k in list(coordnet.state_dict().keys())]
        rf_sapien_rotnet_keys[obj_category] = [f'rotnet.{k}'for k in list(rotnet.state_dict().keys())]
    return rf_sapien_rotnet_keys, rf_sapien_coordnet_keys

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Convert offical checkpoint file to rfvision-style.')
    arg_parser.add_argument('--checkpoint_root', required=True, type=str, help='offical checkpoint files root.')
    args = arg_parser.parse_args()
    offical_ckpt_root = args.checkpoint_root
    rf_sapien_rotnet_keys, rf_sapien_coordnet_keys = get_sapien_keys()
    dirs = [
        'drawers_coord',
        'drawers_rot',
        'glasses_coord',
        'glasses_rot',
        'laptop_coord',
        'laptop_rot',
        'scissors_coord',
        'scissors_rot',
    ]


    for i in dirs:
        ckpt_path = join(offical_ckpt_root, i, 'ckpt/model_0000.pt')
        out_path = join(offical_ckpt_root, i, 'ckpt/rf_model_0000.pt')
        obj_category, net_type = i.split('_')
        if net_type == 'coord':
            keys = rf_sapien_coordnet_keys[obj_category]
        else:
            keys = rf_sapien_rotnet_keys[obj_category]
        main(ckpt_path, out_path, keys, net_type)