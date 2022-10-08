import os
from os.path import join
from models.CAPTRA.scripts.checkpoint_offical2rf_nocs import main

import argparse

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Convert offical checkpoint file to rfvision-style.')
    arg_parser.add_argument('--checkpoint_root', required=True, type=str, help='offical checkpoint files root.')
    args = arg_parser.parse_args()
    offical_ckpt_root = args.checkpoint_root

    dirs = [
        '2_bowl_coord',
        '4_can_coord',
        '3_camera_coord',
        'nocs_ckpt.tar',
        '6_mug_rot',
        'sapien_ckpt.tar',
        '3_camera_rot',
        '1_bottle_coord',
        '5_laptop_coord',
        '5_laptop_rot',
        '1_bottle_rot',
        '2_bowl_rot',
        '4_can_rot',
        '6_mug_coord'
    ]


    for i in dirs:
        if os.path.isdir(join(offical_ckpt_root, i)):
            ckpt_path = join(offical_ckpt_root, i, 'ckpt/model_0000.pt')
            out_path = join(offical_ckpt_root, i, 'ckpt/rf_model_0000.pt')
            main(ckpt_path, out_path)