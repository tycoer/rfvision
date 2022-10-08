import os
import pickle
import re
import warnings
from collections import defaultdict
from enum import Flag

import cv2
from . import ho3d
from . import hodata
import numpy as np
import torch
from . import ho3dutils
from .hodata import HOdata
from .hoquery import BaseQueries, get_trans_queries
from ..utils import meshutils
from liegroups import SE3, SO3
from manopth import manolayer
from manopth.anchorutils import anchor_load_driver
from PIL import Image
from PIL.Image import FASTOCTREE
from termcolor import cprint
from tqdm import tqdm
from ..utils.visualize.vis_contact_info import view_vertex_contact


class HO3DSynt(ho3d.HO3D):
    def __init__(
        self,
        data_root="data",
        data_split="train",
        njoints=21,
        use_cache=True,
        enable_contact=False,
        filter_no_contact=True,
        filter_thresh=10,
        mini_factor=1.0,
        center_idx=9,
        scale_jittering=0.0,
        center_jittering=0.0,
        block_rot=False,
        max_rot=0.0 * np.pi,
        hue=0.15,
        saturation=0.5,
        contrast=0.5,
        brightness=0.5,
        blur_radius=0.5,
        query=None,
        sides="right",
        # *======== HO3D >>>>>>>>>>>>>>>>>>
        split_mode="objects",
        like_v1=True,
        full_image=True,
        full_sequences=False,
        contact_pad_vertex=True,
        contact_pad_anchor=True,
        contact_range_th=1000.0,
        contact_elasti_th=0.00,
        synt_factor=3,
        # *<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        **kwargs,
    ):
        super().__init__(
            data_root=data_root,
            data_split=data_split,
            njoints=njoints,
            use_cache=use_cache,
            enable_contact=enable_contact,
            filter_no_contact=filter_no_contact,
            filter_thresh=filter_thresh,
            mini_factor=mini_factor,
            center_idx=center_idx,
            scale_jittering=scale_jittering,
            center_jittering=center_jittering,
            block_rot=block_rot,
            max_rot=max_rot,
            hue=hue,
            saturation=saturation,
            contrast=contrast,
            brightness=brightness,
            blur_radius=blur_radius,
            query=query,
            sides=sides,
            split_mode=split_mode,
            like_v1=like_v1,
            full_image=full_image,
            full_sequences=full_sequences,
            contact_pad_vertex=contact_pad_vertex,
            contact_pad_anchor=contact_pad_anchor,
            contact_range_th=contact_range_th,
            contact_elasti_th=contact_elasti_th,
            load_objects_color=False,
        )

        self.name = "HO3D_synthesis"
        self.rand_size = synt_factor

    def _preload(self):
        super()._preload()
        self.cache_path = self.cache_path.replace(".pkl", f"_syntf{self.rand_size}.pkl")

    def load_dataset(self):
        super().load_dataset()
        self.hand_fitting_pose = []
        print("loading hand pose...")
        has_warned = False
        for i in tqdm(range(len(self))):
            try:
                self.hand_fitting_pose.append(self._get_hand_fitting_pose(i))
            except:
                if not has_warned:
                    print(f"{self.name} do not have fitted hand pose in {self.split_mode} split mode!")
                    has_warned = True
                self.hand_fitting_pose.append(np.zeros(48, dtype=np.float32))  # use 0 instead

    def load_seq_frames(self, subfolder=None, seqs=None, trainval_idx=6000):
        """
        trainval_idx (int): How many frames to include in training split when
                using trainval/val/test split
        """
        if self.split_mode == "paper":
            if self.data_split in ["train", "trainval", "val"]:
                info_path = os.path.join(self.root, "train.txt")
                subfolder = "train"
            elif self.data_split == "test":
                info_path = os.path.join(self.root, "evaluation.txt")
                subfolder = "evaluation"
            else:
                assert False
            with open(info_path, "r") as f:
                lines = f.readlines()
            seq_frames = [line.strip().split("/") for line in lines]
            if self.data_split == "trainval":
                seq_frames = seq_frames[:trainval_idx]
            elif self.data_split == "val":
                seq_frames = seq_frames[trainval_idx:]
        elif self.split_mode == "objects":
            root = self.root.replace(self.name, "HO3D")
            seq_frames = []
            for seq in sorted(seqs):
                seq_folder = os.path.join(root, subfolder, seq)
                meta_folder = os.path.join(seq_folder, "meta")
                img_nb = len(os.listdir(meta_folder))
                for img_idx in range(img_nb):
                    seq_frames.append([seq, f"{img_idx:04d}"])
        else:
            assert False
        return seq_frames, subfolder

    def load_contact_infos(self, idxs, seq_map):
        contact_info = []
        for i in range(len(idxs)):
            seq, idx = idxs[i]
            path = seq_map[seq][idx]["img"]
            path = path.replace(self.root, self.root_supp)
            path = path.replace("_synthesis", "")  # load no synt contact info
            path = re.sub(r"_\d", "", path)  # remove rand mark like '_1' except ho3dgen
            path = path.replace("rgb", "contact_info")
            path = path.replace("png", "pkl")
            contact_info.append(path)
        return contact_info

    def load_annots(self, obj_meshes={}, seq_frames=[], subfolder="train"):
        """
        Args:
            split (str): HO3DV2 split in [train|trainval|val|test]
                train = trainval U(nion) val
            rand_size (int): synthetic data counts
                will be 0 if you want to use the vanilla data
        """

        vhand_path = os.path.join(self.root_extra_info, "hand_palm_full.txt")
        vid = np.loadtxt(vhand_path, dtype=np.int)

        idxs = []
        seq_map = defaultdict(list)
        seq_counts = defaultdict(int)
        for idx_count, (seq, frame_idx) in enumerate(tqdm(seq_frames)):
            if int(frame_idx) % round(self.mini_factor) != 0:
                continue
            seq_folder = os.path.join(self.root, subfolder, seq)
            meta_folder = os.path.join(seq_folder, "meta")
            rgb_folder = os.path.join(seq_folder, "rgb")

            meta_path = os.path.join(meta_folder, f"{frame_idx}.pkl")

            meta_path = meta_path.replace(self.name, "HO3D")

            with open(meta_path, "rb") as p_f:
                annot = pickle.load(p_f)
                if annot["handJoints3D"].size == 3:
                    annot["handTrans"] = annot["handJoints3D"]
                    annot["handJoints3D"] = annot["handJoints3D"][np.newaxis, :].repeat(21, 0)
                    annot["handPose"] = np.zeros(48, dtype=np.float32)
                    annot["handBeta"] = np.zeros(10, dtype=np.float32)

            # filter no contact
            if self.filter_no_contact and ho3dutils.min_contact_dis(annot, obj_meshes, vid) > self.filter_thresh:
                continue

            # ? synthetic data
            for i in range(self.rand_size):

                rand_meta_path = os.path.join(meta_folder, f"{frame_idx}_{i}.pkl")
                # jump if file not synt
                if not os.path.isfile(rand_meta_path):
                    continue
                with open(rand_meta_path, "rb") as p_f:
                    rand_annot = pickle.load(p_f)
                    rand_annot.update(annot)

                    img_path = os.path.join(rgb_folder, f"{frame_idx}_{i}.png")
                    rand_annot["img"] = img_path
                    rand_annot["frame_idx"] = f"{frame_idx}_{i}"

                    seq_map[seq].append(rand_annot)
                    idxs.append((seq, seq_counts[seq]))
                    seq_counts[seq] += 1

        return seq_map, idxs

    def _get_hand_fitting_pose(self, idx):
        path = self.get_image_path(idx)
        path = path.replace("rgb", "meta")
        path = path.replace(".png", ".npy")
        pose = np.load(path)
        return pose.astype(np.float32)

    def get_hand_pose_wrt_cam(self, idx):
        return self.hand_fitting_pose[idx]

    def _get_rand_rot_tsl(self, idx):
        annot = self.get_annot(idx)
        rand_rot = SO3.exp(annot["rr"]).as_matrix()
        rand_tsl = annot["rt"]
        return rand_rot, rand_tsl

    def get_joints3d(self, idx):
        joint3d = super().get_joints3d(idx)
        obj_transf = super().get_obj_transf_wrt_cam(idx)
        obj_rot = obj_transf[:3, :3]  # (3, 3)
        obj_tsl = obj_transf[:3, 3:]  # (3, 1)
        rand_rot, rand_tsl = self._get_rand_rot_tsl(idx)
        joint3d = joint3d - obj_tsl.T
        joint3d = np.linalg.inv(obj_rot).dot(joint3d.T).T
        rand_joint = rand_rot.dot(joint3d.T).T + rand_tsl
        rand_joint = (obj_rot.dot(rand_joint.T) + obj_tsl).T
        rand_joint = np.array(rand_joint).astype(np.float32)
        return rand_joint

    def get_hand_verts3d(self, idx):
        hand_verts = super().get_hand_verts3d(idx)
        # return hand_verts
        obj_transf = super().get_obj_transf_wrt_cam(idx)
        obj_rot = obj_transf[:3, :3]  # (3, 3)
        obj_tsl = obj_transf[:3, 3:]  # (3, 1)
        rand_rot, rand_tsl = self._get_rand_rot_tsl(idx)
        hand_verts = hand_verts - obj_tsl.T
        hand_verts = np.linalg.inv(obj_rot).dot(hand_verts.T).T
        rand_hand = rand_rot.dot(hand_verts.T).T + rand_tsl
        rand_hand = (obj_rot.dot(rand_hand.T) + obj_tsl).T
        rand_hand = np.array(rand_hand).astype(np.float32)
        return rand_hand

    def get_obj_transf_wrt_cam(self, idx):
        rand_rot, rand_tsl = self._get_rand_rot_tsl(idx)
        rand_transf = np.concatenate([rand_rot, rand_tsl[:, np.newaxis]], axis=1)
        rand_transf = np.concatenate([rand_transf, np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)
        return super().get_obj_transf_wrt_cam(idx).dot(rand_transf)

    def get_obj_verts_transf(self, idx):
        obj_can = self.get_obj_verts_can(idx)[0]
        obj_can = np.concatenate([obj_can, np.ones((obj_can.shape[0], 1))], axis=1)
        return self.get_obj_transf_wrt_cam(idx).dot(obj_can.T).T[:, :3].astype(np.float32)

    def get_obj_corners3d(self, idx):
        obj_corners = self.get_obj_corners_can(idx)
        obj_corners = np.concatenate([obj_corners, np.ones((obj_corners.shape[0], 1))], axis=1)
        obj_corners = self.get_obj_transf_wrt_cam(idx).dot(obj_corners.T).T[:, :3].astype(np.float32)
        return obj_corners


def main(args):
    import cv2

    ho_dataset = HOdata.get_dataset(
        dataset="ho3dsynt",
        data_root="data",
        data_split=args.data_split,
        split_mode="official",
        use_cache=True,
        mini_factor=1,
        center_idx=9,
        enable_contact=True,
        like_v1=True,
        filter_no_contact=False,
        filter_thresh=10,
        block_rot=True,
        synt_factor=3,
    )
    import pickle

    import prettytable as pt
    from .ho3d import HO3D

    # def fit_transf(raw_joints, target_joints):
    #     raw_joints = np.concatenate([raw_joints, np.ones((raw_joints.shape[0], 1), dtype=np.float32)], axis=1)
    #     def ls_loss(se3):
    #         transf = SE3.exp(se3).as_matrix()
    #         joints = (transf @ raw_joints.T).T[:, :3]
    #         res = joints.ravel() - target_joints.ravel()
    #         return res
    #     return least_squares(ls_loss, np.ones(6, dtype=np.float32)).x
    # hand_poses = []
    # hand_tsls = []
    # cam_extr = raw_dataset.cam_extr[:3, :3]
    # for i in tqdm(range(len(ho_dataset))):
    #     raw_joints = (cam_extr @ raw_dataset.get_joints3d(i // 3).T).T
    #     assert np.allclose(raw_dataset.get_joints3d(i // 3), HO3D.get_joints3d(ho_dataset, i))
    #     target_joints = (cam_extr @ ho_dataset.get_joints3d(i).T).T
    #     res = fit_transf(raw_joints, target_joints)
    #     raw_annot = raw_dataset.get_annot(i // 3)
    #     raw_pose = raw_annot["handPose"]
    #     raw_tsl = raw_annot["handTrans"]
    #     transf = SE3.exp(res).as_matrix()
    #     # print(transf)
    #     rot = transf[:3, :3]
    #     tsl = transf[:3, 3].T
    #     raw_root_rot = SO3.exp(raw_pose[:3]).as_matrix()
    #     hand_root_rot = SO3.log(SO3.from_matrix(cam_extr @ rot @ raw_root_rot, normalize=True))
    #     hand_pose = np.concatenate([hand_root_rot, raw_pose[3:]], axis=0)
    #     hand_tsl = (cam_extr @ (raw_tsl + tsl).T).T
    #     hand_poses.append(hand_pose.astype(np.float32))
    #     hand_tsls.append(hand_tsl.astype(np.float32))
    #     # print(ho_dataset.get_image_path(i).replace("rgb", "meta").replace("png", "npy"))
    #     with open(ho_dataset.get_image_path(i).replace("rgb", "meta").replace("png", "npy"), "wb") as f:
    #         np.save(f, hand_pose.astype(np.float32))
    #     # if i > 10:
    #     #     break
    # assert False
    from hocontact.utils.logger import logger
    from scipy.optimize._lsq.least_squares import least_squares

    # for i in range(len(raw_dataset)):
    #     for j in range(3):
    #         assert np.allclose(raw_dataset.get_obj_transf_wrt_cam(i), HO3D.get_obj_transf_wrt_cam(ho_dataset, i * 3 + j))
    # # print(raw_dataset.get_obj_transf_wrt_cam(2488))
    # # print(ho_dataset.get_hand_pose_wrt_cam(0, raw_dataset))
    # assert False
    from tqdm import tqdm

    idx = np.random.randint(len(ho_dataset))

    cprint(len(ho_dataset), "cyan")

    sample = ho_dataset[idx]
    tb = pt.PrettyTable(padding_width=3, header=False)
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            tb.add_row([key, type(value), value.shape])
        elif isinstance(value, torch.Tensor):
            tb.add_row([key, type(value), tuple(value.size())])
        else:
            tb.add_row([key, type(value), value])
    print(f"{'=' * 40} ALL FHB SAMPLE KEYS {'>' * 40}", "blue")
    print(str(tb))

    if args.render:
        view_vertex_contact(ho_dataset)

    if args.vis:

        def view_data(ho_dataset):
            for i in range(len(ho_dataset)):
                # i = len(ho_dataset) - 1 - i
                i = np.random.randint(0, len(ho_dataset))
                objverts2d = ho_dataset.get_obj_verts2d(i)
                joint2d = ho_dataset.get_joints2d(i)

                # TEST: obj_transf @ obj_verts_can == obj_verts_transf >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                obj_transf = ho_dataset.get_obj_transf_wrt_cam(i)
                obj_rot = obj_transf[:3, :3]  # (3, 3)
                obj_tsl = obj_transf[:3, 3:]  # (3, 1)

                obj_verts_can, _, __ = ho_dataset.get_obj_verts_can(i)  # (N, 3)
                obj_verts_pred = (obj_rot.dot(obj_verts_can.transpose()) + obj_tsl).transpose()
                obj_verts2d_pred = ho_dataset.project(obj_verts_pred, ho_dataset.get_cam_intr(i))

                # TEST: obj_transf @ obj_corners_can == obj_corners_3d >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                obj_corners_can = ho_dataset.get_obj_corners_can(i)
                obj_corners_transf = (obj_rot.dot(obj_corners_can.transpose()) + obj_tsl).transpose()
                obj_corners2d = ho_dataset.project(obj_corners_transf, ho_dataset.get_cam_intr(i))

                # TEST: MANO(get_hand_pose_wrt_cam) + get_hand_tsl_wrt_cam == get_hand_verts3d >>>>>>>>>>>>>>>>>>>>>>>>>>
                # hand_pose = torch.from_numpy(ho_dataset.get_hand_pose_wrt_cam(i)).unsqueeze(0)
                hand_pose = torch.from_numpy(ho_dataset.get_hand_pose_wrt_cam(i)).unsqueeze(0)
                hand_shape = torch.from_numpy(ho_dataset.get_hand_shape(i)).unsqueeze(0)
                hand_tsl = ho_dataset.get_hand_tsl_wrt_cam(i)

                hand_verts, hand_joints = ho_dataset.layer(hand_pose, hand_shape)
                hand_verts = np.array(hand_verts.squeeze(0)) + hand_tsl
                hand_verts_2d = ho_dataset.project(hand_verts, ho_dataset.get_cam_intr(i))

                hand_verts_2dgt = ho_dataset.get_hand_verts2d(i)
                img = ho_dataset.get_image(i)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for j in range(obj_verts2d_pred.shape[0]):
                    v = obj_verts2d_pred[j]
                    cv2.circle(img, (v[0], v[1]), radius=2, thickness=-1, color=(0, 255, 0))
                for j in range(joint2d.shape[0]):
                    v = joint2d[j]
                    cv2.circle(img, (v[0], v[1]), radius=3, thickness=-1, color=(0, 255, 255))
                for j in range(hand_verts_2dgt.shape[0]):
                    v = hand_verts_2dgt[j]
                    cv2.circle(img, (v[0], v[1]), radius=2, thickness=-1, color=(255, 255, 0))
                for j in range(hand_verts_2d.shape[0]):
                    v = hand_verts_2d[j]
                    cv2.circle(img, (v[0], v[1]), radius=1, thickness=-1, color=(255, 0, 0))
                for j in range(obj_corners2d.shape[0]):
                    v = obj_corners2d[j]
                    cv2.circle(img, (v[0], v[1]), radius=8, thickness=-1, color=(0, 0, 255))
                cv2.imshow("ho3d", img)
                cv2.waitKey(0)

        view_data(ho_dataset)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="test fhbdataset")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--data_split", choices=["train", "test", "all"], default="train", type=str)
    parser.add_argument("--split_mode", choices=["", "objects"], default="objects", type=str)
    parser.add_argument("--synt_factor", type=int, choices=[1, 2, 3], default=1)
    main(parser.parse_args())
