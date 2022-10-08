import torch
import pickle
import numpy as np
from copy import deepcopy

from .captra_backbone import PartCanonNet, CoordNet
from os.path import join as pjoin
from models.CAPTRA.utils.utils import Timer, add_dict, divide_dict, log_loss_summary, get_ith_from_batch, \
    cvt_torch, ensure_dirs, per_dict_to_csv

from models.CAPTRA.utils.part_dof_utils import part_model_batch_to_part, eval_part_full, add_noise_to_part_dof
from models.CAPTRA.utils.bbox_utils import get_pred_nocs_corners, eval_single_part_iou, tensor_bbox_from_corners, yaxis_from_corners
from models.CAPTRA.utils.nocs_data_process import full_data_from_depth_image
from models.CAPTRA.models.loss import compute_miou_loss, compute_nocs_loss, choose_coord_by_label, \
    compute_part_dof_loss, rot_trace_loss, compute_point_pose_loss, rot_yaxis_loss


class CaptraTrackModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.rotnet = PartCanonNet(**cfg['rotnet'])
        self.coordnet = CoordNet(**cfg['coordnet'])
        self.num_parts = int(cfg['obj_cfg']['num_parts'])
        self.tree = cfg['obj_cfg']['tree']
        self.root = [p for p in range(len(self.tree)) if self.tree[p] == -1][0]
        self.gt_init = cfg['init_frame']['gt']
        self.nocs_otf = 'nocs_otf' in cfg and cfg['nocs_otf']
        if self.nocs_otf:
            assert cfg['batch_size'] == 1
            self.meta_root = pjoin(cfg['root_dset'], 'render', 'real_test', cfg['obj_category'])
        self.radius = cfg['data_radius']
        self.cfg = cfg
        self.track_cfg = cfg['track_cfg']
        self.timer = Timer(True)
        self.pose_perturb_cfg = cfg['pose_perturb_cfg']
        self.pose_perturb_cfg['rotation'] = np.deg2rad(self.pose_perturb_cfg['rotation'])
        self.sym = cfg['obj_cfg']['sym']
        self.to(cfg['device'])
        self.device = cfg['device']

    def convert_init_frame_data(self, frame):
        feed_frame = {}
        for key, item in frame.items():
            if key not in ['meta', 'labels', 'points', 'nocs']:
                continue
            if key in ['meta']:
                pass
            elif key in ['labels']:
                item = item.long().to(self.device)
            else:
                item = item.float().to(self.device)
            feed_frame[key] = item
        gt_part = part_model_batch_to_part(cvt_torch(frame['meta']['nocs2camera'], self.device), self.num_parts,
                                           self.device)
        feed_frame.update({'gt_part': gt_part})

        return feed_frame

    def convert_subseq_frame_data(self, data):
        gt_part = part_model_batch_to_part(cvt_torch(data['meta']['nocs2camera'], self.device), self.num_parts,
                                           self.device)
        input = {'points': data['points'],
                 'points_mean': data['meta']['points_mean'],
                 'gt_part': gt_part}

        if 'nocs' in data:
            input['npcs'] = data['nocs']
        input = cvt_torch(input, self.device)
        input['meta'] = data['meta']
        if 'labels' in data:
            input['labels'] = data['labels'].long().to(self.device)
        return input

    def convert_subseq_frame_npcs_data(self, data):
        input = {}
        for key, item in data.items():
            if key not in ['meta', 'labels', 'points', 'nocs']:
                continue
            elif key in ['meta']:
                pass
            elif key in ['labels']:
                item = item.long().to(self.device)
            else:
                item = item.float().to(self.device)
            input[key] = item
        input['points_mean'] = data['meta']['points_mean'].float().to(self.device)
        return input

    def set_data(self, data):
        feed_dict = []
        npcs_feed_dict = []
        for i, frame in enumerate(data):
            if i == 0:
                feed_dict.append(self.convert_init_frame_data(frame))
            else:
                feed_dict.append(self.convert_subseq_frame_data(frame))
            npcs_feed_dict.append(self.convert_subseq_frame_npcs_data(frame))
        return feed_dict, npcs_feed_dict

    def forward(self,
                data,
                no_eval=False,
                save_dir='../runs/results/data'):
        feed_dict, npcs_feed_dict = self.set_data(data)
        self.timer.tick()
        pred_poses = []
        gt_part = feed_dict[0]['gt_part']
        if self.gt_init:
            pred_poses.append(gt_part)
        else:
            part = add_noise_to_part_dof(gt_part, self.pose_perturb_cfg)
            if 'crop_pose' in feed_dict[0]['meta']:
                crop_pose = part_model_batch_to_part(cvt_torch(feed_dict[0]['meta']['crop_pose'], self.device),
                                                     self.num_parts, self.device)
                for key in ['translation', 'scale']:
                    part[key] = crop_pose[key]
            pred_poses.append(part)

        self.timer.tick()

        time_dict = {'crop': 0.0, 'npcs_net': 0.0, 'rot_all': 0.0}

        frame_nums = []
        npcs_pred = []
        with torch.no_grad():
            for i, input in enumerate(feed_dict):
                frame_nums.append([path.split('.')[-2].split('/')[-1] for path in input['meta']['path']])
                if i == 0:
                    npcs_pred.append(None)
                    continue
                perturbed_part = add_noise_to_part_dof(feed_dict[i - 1]['gt_part'], self.pose_perturb_cfg)
                if 'crop_pose' in feed_dict[i]['meta']:
                    crop_pose = part_model_batch_to_part(
                        cvt_torch(feed_dict[i]['meta']['crop_pose'], self.device),
                        self.num_parts, self.device)
                    for key in ['translation', 'scale']:
                        perturbed_part[key] = crop_pose[key]

                last_pose = {key: value.clone() for key, value in pred_poses[-1].items()}

                self.timer.tick()
                if self.nocs_otf:
                    center = last_pose['translation'].reshape(3).detach().cpu().numpy()  # [3]
                    scale = last_pose['scale'].reshape(1).detach().cpu().item()
                    depth_path = input['meta']['ori_path'][0]
                    category, instance = input['meta']['path'][0].split('/')[-4:-2]
                    pre_fetched = input['meta']['pre_fetched']
                    pre_fetched = {key: value.reshape(value.shape[1:]) for key, value in pre_fetched.items()}

                    pose = {key: value.squeeze(0).squeeze(0).detach().cpu().numpy() for key, value in
                            input['gt_part'].items()}
                    full_data = full_data_from_depth_image(depth_path, category, instance, center, self.radius * scale,
                                                           pose,
                                                           num_points=input['points'].shape[-1], device=self.device,
                                                           mask_from_nocs2d=self.track_cfg['nocs2d_label'],
                                                           nocs2d_path=self.track_cfg['nocs2d_path'],
                                                           pre_fetched=pre_fetched)

                    points, nocs, labels = full_data['points'], full_data['nocs'], full_data['labels']

                    points = cvt_torch(points, self.device)
                    points -= npcs_feed_dict[i]['points_mean'].reshape(1, 3)
                    input['points'] = points.transpose(-1, -2).reshape(1, 3, -1)
                    input['labels'] = torch.tensor(full_data['labels']).to(self.device).long().reshape(1, -1)
                    nocs = cvt_torch(nocs, self.device)
                    npcs_feed_dict[i]['points'] = input['points']
                    npcs_feed_dict[i]['labels'] = input['labels']
                    npcs_feed_dict[i]['nocs'] = nocs.transpose(-1, -2).reshape(1, 3, -1)

                    time_dict['crop'] += self.timer.tick()

                state = {'part': last_pose}
                input['state'] = state

                npcs_canon_pose = {key: last_pose[key][:, self.root].clone() for key in
                                   ['rotation', 'translation', 'scale']}
                npcs_input = npcs_feed_dict[i]
                npcs_input['canon_pose'] = npcs_canon_pose
                npcs_input['init_part'] = last_pose
                cur_npcs_pred = self.coordnet(npcs_input)  # seg: [B, P, N], npcs: [B, P * 3, N]
                npcs_pred.append(cur_npcs_pred)
                pred_npcs, pred_seg = cur_npcs_pred['nocs'], cur_npcs_pred['seg']
                pred_npcs = pred_npcs.reshape(len(pred_npcs), self.num_parts, 3, -1)  # [B, P, 3, N]
                pred_labels = torch.max(pred_seg, dim=-2)[1]  # [B, P, N] -> [B, N]

                time_dict['npcs_net'] += self.timer.tick()

                input['pred_labels'], input['pred_nocs'] = pred_labels, pred_npcs
                input['pred_label_conf'] = pred_seg[:, 0]  # [B, P, N]
                if self.track_cfg['gt_label'] or self.track_cfg['nocs2d_label']:
                    input['pred_labels'] = npcs_input['labels']

                pred_dict = self.rotnet(input, test_mode=True)
                pred_poses.append(pred_dict['part'])

                time_dict['rot_all'] += self.timer.tick()

        pred_dict = {'poses': pred_poses, 'npcs_pred': npcs_pred}

        loss_dict = {}
        if not no_eval:
            loss_dict = self.compute_loss(feed_dict=feed_dict,
                                          npcs_feed_dict=npcs_feed_dict,
                                          pred_dict=pred_dict)

        if save_dir is not None:
            gt_corners = feed_dict[0]['meta']['nocs_corners'].cpu().numpy()
            corner_list = []
            for i, pred_pose in enumerate(pred_dict['poses']):
                if i == 0:
                    corner_list.append(None)
                    continue
                pred_labels = torch.max(pred_dict['npcs_pred'][i]['seg'], dim=-2)[1]  # [B, P, N] -> [B, N]
                pred_nocs = choose_coord_by_label(pred_dict['npcs_pred'][i]['nocs'].transpose(-1, -2),
                                                  pred_labels)
                pred_corners = get_pred_nocs_corners(pred_labels, pred_nocs, self.num_parts)
                corner_list.append(pred_corners)

            gt_poses = [{key: value.detach().cpu().numpy() for key, value in frame[f'gt_part'].items()}
                        for frame in feed_dict]
            save_dict = {'pred': {'poses': [{key: value.detach().cpu().numpy() for key, value in pred_pose.items()}
                                            for pred_pose in pred_poses],
                                  'corners': corner_list},
                         'gt': {'poses': gt_poses, 'corners': gt_corners},
                         'frame_nums': frame_nums}

            # save_path = pjoin(self.cfg['experiment_dir'], 'results', 'data')
            save_path = save_dir
            ensure_dirs([save_path])
            for i, path in enumerate(feed_dict[0]['meta']['path']):
                instance, track_num = path.split('.')[-2].split('/')[-3: -1]
                with open(pjoin(save_path, f'{instance}_{track_num}.pkl'), 'wb') as f:
                    cur_dict = get_ith_from_batch(save_dict, i, to_single=False)
                    pickle.dump(cur_dict, f)
            print(f"Predictions are saved to {pjoin(save_path, f'{instance}_{track_num}.pkl')}")
        return pred_dict, loss_dict

    def compute_loss(self, feed_dict, pred_dict, npcs_feed_dict, per_instance=False, eval_iou=False):
        loss_dict = {}

        avg_pose_diff, all_pose_diff = {}, {}
        avg_init_diff, all_init_diff = {}, {}
        avg_iou, all_iou = {}, {}
        avg_seg_loss, all_seg_loss = [], {}
        avg_nocs_loss, all_nocs_loss = [], {}
        gt_corners = feed_dict[0]['meta']['nocs_corners'].float().to(self.device)

        for i, pred_pose in enumerate(pred_dict['poses']):

            pose_diff, per_diff = eval_part_full(feed_dict[i]['gt_part'], pred_pose,
                                                 per_instance=per_instance, yaxis_only=self.sym)

            if i > 0:
                add_dict(avg_pose_diff, pose_diff)
                if per_instance:
                    self.record_per_diff(feed_dict[i], per_diff)

            all_pose_diff[i] = deepcopy(pose_diff)

            if i > 0:
                init_pose_diff, init_per_diff = eval_part_full(feed_dict[i]['gt_part'], pred_dict['poses'][i - 1],
                                                               per_instance=per_instance,
                                                               yaxis_only=self.sym)

                add_dict(avg_init_diff, init_pose_diff)
                all_init_diff[i] = deepcopy(init_pose_diff)

            if i > 0:
                if 'labels' in npcs_feed_dict[i]:
                    seg_loss = compute_miou_loss(pred_dict['npcs_pred'][i]['seg'], npcs_feed_dict[i]['labels'],
                                                 per_instance=False)
                    avg_seg_loss.append(seg_loss)
                    all_seg_loss[i] = seg_loss

                pred_labels = torch.max(pred_dict['npcs_pred'][i]['seg'], dim=-2)[1]  # [B, P, N] -> [B, N]

                if 'nocs' in npcs_feed_dict[i]:
                    nocs_loss = compute_nocs_loss(pred_dict['npcs_pred'][i]['nocs'], npcs_feed_dict[i]['nocs'],
                                                  labels=pred_labels,
                                                  confidence=None, loss='l2', self_supervise=False,
                                                  per_instance=False)
                    avg_nocs_loss.append(nocs_loss)
                    all_nocs_loss[i] = nocs_loss

                pred_nocs = choose_coord_by_label(pred_dict['npcs_pred'][i]['nocs'].transpose(-1, -2),
                                                  pred_labels)

                if eval_iou:
                    pred_corners = get_pred_nocs_corners(pred_labels, pred_nocs, self.num_parts)
                    pred_corners = torch.tensor(pred_corners).to(self.device).float()

                    def calc_iou(gt_pose, pred_pose):
                        iou, per_iou = eval_single_part_iou(gt_corners, pred_corners, gt_pose, pred_pose,
                                                            separate='both', nocs=self.nocs_otf, sym=self.sym)

                        return iou, per_iou

                    iou, per_iou = calc_iou(feed_dict[i]['gt_part'], pred_pose)
                    add_dict(avg_iou, iou)
                    if per_instance:
                        self.record_per_diff(feed_dict[i], per_iou)
                    all_iou[i] = deepcopy(iou)

        avg_pose_diff = divide_dict(avg_pose_diff, len(pred_dict['poses']) - 1)
        avg_init_diff = divide_dict(avg_init_diff, len(pred_dict['poses']) - 1)
        loss_dict.update({'avg_pred': avg_pose_diff, 'avg_init': avg_init_diff,
                          'frame_pred': all_pose_diff, 'frame_init': all_init_diff})
        if len(avg_seg_loss) > 0:
            avg_seg_loss = torch.mean(torch.stack(avg_seg_loss))
            loss_dict.update({'avg_seg': avg_seg_loss, 'frame_seg': all_seg_loss})
        if len(avg_nocs_loss) > 0:
            avg_nocs_loss = torch.mean(torch.stack(avg_nocs_loss))
            loss_dict.update({'avg_nocs': avg_nocs_loss, 'frame_seg': all_nocs_loss})
        if eval_iou:
            avg_iou = divide_dict(avg_iou, len(pred_dict['poses']) - 1)
            loss_dict.update({'avg_iou': avg_iou, 'frame_iou': all_iou})

        return loss_dict

