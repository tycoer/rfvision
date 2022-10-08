from mmdet3d.models.builder import build_detector
from mmdet3d.datasets.builder import build_dataset
from mmcv import Config
from mmcv.runner import load_checkpoint
from torch.utils.data.dataloader import DataLoader
from models.CAPTRA.models.captra_trackmodel import CaptraTrackModel
from models.CAPTRA.datasets.captra_dataset import SequenceData
from models.CAPTRA.utils.utils import add_dict, log_loss_summary
import argparse
import numpy as np
from tqdm import tqdm
import torch
import time


nocs_obj_cfg = {'1': {'name': 'bottle',
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

sapien_obj_cfg =  {'glasses': {'sym': False,
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

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Track object pose in scenes.')
    arg_parser.add_argument('--rotnet_checkpoint', required=True, type=str, help='checkpoint file path.',
                            default='/home/hanyang/CAPTRA-main/6_mug_rot/ckpt/model_0000.pt')
    arg_parser.add_argument('--coordnet_checkpoint', required=True, type=str, help='checkpoint file path.',
                            default='/home/hanyang/CAPTRA-main/6_mug_rot/ckpt/model_0000.pt')
    # arg_parser.add_argument('--config', type=str, help='config file path.', required=True)
    arg_parser.add_argument('--obj_category', type=str, choices=['1', '2', '3', '4', '5', '6',
                                                                 'drawers', 'glasses', 'scissors', 'laptop'], default='1')
    arg_parser.add_argument('--mode', type=str, choices=['real_test', 'train', 'val', 'bmvc', 'test_seq'], default='real_test')
    arg_parser.add_argument('--save_dir', type=str, default='../runs/1_bottle_rot/results/data')
    arg_parser.add_argument('--no_eval', action='store_true', default=False)
    arg_parser.add_argument('--data_root', type=str, default='../data/sapien_data')
    arg_parser.add_argument('--dataset_type', type=str, choices=['nocs', 'sapien'], default='nocs')

    args = arg_parser.parse_args()

    rotnet_ckpt_path = args.rotnet_checkpoint
    coordnet_ckpt_path = args.coordnet_checkpoint
    obj_category = args.obj_category
    mode = args.mode
    save_dir = args.save_dir
    no_eval = args.no_eval
    data_root = args.data_root
    nocs_data = True if args.dataset_type == 'nocs' else False
    obj_cfg = nocs_obj_cfg if args.dataset_type == 'nocs' else sapien_obj_cfg

    data_cfg = dict(obj_cfg=obj_cfg[obj_category],
                    obj_category=obj_category,
                    basepath=data_root,
                    num_points=4096,
                    dataset_length=None,
                    synthetic=True,
                    nocs_data=nocs_data,
                    num_expr='1_bottle_coord_new',
                    data_radius=0.6,
                    pose_perturb={'type': 'normal', 's': 0.02, 'r': 5.0, 't': 0.03},
                    num_frames=100,
                    )

    model_cfg = dict(
        obj_cfg=obj_cfg[obj_category],
        coordnet=dict(
            nocs_head_dims=[128],
            obj_cfg=obj_cfg[obj_category],
            backbone_out_dim=128,
            extra_dims=1,
        ),
        rotnet=dict(
            obj_cfg=obj_cfg[obj_category],
            backbone_out_dim=128,
            network_type='rot_coord_track',

        ),
        track_cfg=dict(gt_label=False,
                       nocs2d_label=False,
                       nocs2d_path=None),
        init_frame=dict(gt=False),
        nocs_otf=True,
        batch_size=1,
        root_dset='/disk1/data/captar',
        obj_category='1',
        data_radius=0.6,
        pose_perturb_cfg=dict(type='normal',
                              scale=0.02,
                              translation=0.03,
                              rotation=5.0),
        device=0  # 目前只支持device=0, 这是mmdet3d中pointnet2的一个bug, mmdet3d中的sa模块只能在0号卡上
    )

    assert model_cfg['device'] == 0, "Only device=0 supported."
    model = CaptraTrackModel(cfg=model_cfg)
    model.eval()

    rotnet_ckpt = load_checkpoint(model, rotnet_ckpt_path)
    coordnet_ckpt = load_checkpoint(model, coordnet_ckpt_path)


    dataset = SequenceData(cfg=data_cfg, mode=mode)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

    zero_time = time.time()
    time_dict = {'data_proc': 0.0, 'network': 0.0}
    total_frames = 0
    test_loss = {'cnt': 0}
    torch.multiprocessing.set_start_method('spawn')

    for i, data in enumerate(tqdm(dataloader)):
        num_frames = len(data)
        total_frames += num_frames
        print(f'Trajectory {i}, {num_frames:8} frames****************************')

        start_time = time.time()
        elapse = start_time - zero_time
        time_dict['data_proc'] += elapse
        print(f'Data Preprocessing: {elapse:8.2f}s {num_frames / elapse:8.2f}FPS')
        with torch.no_grad():
            pred_dict, loss_dict = model(data, save_dir = save_dir, no_eval = no_eval)

        elapse = time.time() - start_time
        time_dict['network'] += elapse
        print(f'Network Forwarding: {elapse:8.2f}s {num_frames / elapse:8.2f}FPS')
        loss_dict['cnt'] = 1
        add_dict(test_loss, loss_dict)

        zero_time = time.time()

    print(f'Overall, {total_frames:8} frames****************************')
    print(f'Data Preprocessing: {time_dict["data_proc"]:8.2f}s {total_frames / time_dict["data_proc"]:8.2f}FPS')
    print(f'Network Forwarding: {time_dict["network"]:8.2f}s {total_frames / time_dict["network"]:8.2f}FPS')
    if model_cfg['batch_size'] > 1:
        print(f'PLEASE SET batch_size = 1 TO TEST THE SPEED. CURRENT BATCH_SIZE: cfg["batch_size"]')

    # cnt = test_loss.pop('cnt')
    # log_loss_summary(test_loss, cnt, lambda x, y: log_string('Test {} is {}'.format(x, y)))
    # if save and not no_eval:
    #     trainer.model.save_per_diff()