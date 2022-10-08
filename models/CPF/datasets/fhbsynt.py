import os

import numpy as np
import torch
from liegroups import SO3

from .fhb import FPHB
from .hodata import HOdata


class FPHBSynt(FPHB):
    def __init__(
        self,
        data_root="data",
        data_split="train",
        split_mode="actions",
        njoints=21,
        use_cache=True,
        filter_no_contact=True,
        filter_thresh=10,  # mm
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
        # *======== FHB >>>>>>>>>>>>>>>>>>>
        full_image=True,
        reduce_res=True,
        enable_contact=False,
        contact_pad_vertex=True,
        contact_pad_anchor=True,
        contact_range_th=1000.0,
        contact_elasti_th=0.00,
        synt_factor=3,
        # *<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        **kwargs,
    ):
        super().__init__(
            data_root,
            data_split,
            split_mode,
            njoints,
            use_cache,
            filter_no_contact,
            filter_thresh,
            mini_factor,
            center_idx,
            scale_jittering,
            center_jittering,
            block_rot,
            max_rot,
            hue,
            saturation,
            contrast,
            brightness,
            blur_radius,
            query,
            sides,
            full_image,
            reduce_res,
            enable_contact,
            contact_pad_vertex,
            contact_pad_anchor,
            contact_range_th,
            contact_elasti_th,
        )
        self.name = "fhbhands_synthesis"
        self.rand_size = synt_factor

    def _preload(self):
        # ! use these path CAREFULLY, may led to unexpected bugs
        self.super_name = "fhbhands"
        self.root = os.path.join(self.data_root, self.name)
        self.root_supp = os.path.join(self.data_root, f"{self.name}_supp")
        self.root_extra_info = os.path.normpath("assets")
        self.info_root = os.path.join(self.root.replace("_synthesis", ""), "Subjects_info")
        self.info_split = os.path.join(self.root.replace("_synthesis", ""), "data_split_action_recognition.txt")
        small_rgb_root = os.path.join(self.root.replace("_synthesis", ""), "Video_files_480")
        if os.path.exists(small_rgb_root) and self.reduce_res:
            self.rgb_root = small_rgb_root
            self.reduce_factor = 1 / 4
        else:
            self.rgb_root = os.path.join(self.root.replace("_synthesis", ""), "Video_files")
            self.reduce_factor = 1
        self.skeleton_root = os.path.join(self.root.replace("_synthesis", ""), "Hand_pose_annotation_v1")

        self.rgb_template = "color_{:04d}.jpeg"
        # Joints are numbered from tip to base, we want opposite

        self.cache_path = os.path.join("common", "cache", self.name)

        # NOTE: eci for "enable contact info"
        self.cache_path = os.path.join(
            self.cache_path,
            f"{self.data_split}_{self.split_mode}_mf{self.mini_factor}"
            f"_rf{self.reduce_factor}"
            f"_fct{self.filter_thresh if self.filter_no_contact else '(x)'}"
            f"_ec{'' if self.enable_contact else '(x)'}"
            f"_syntf{self.rand_size}"
            f".pkl",
        )

    def _get_rand_rot_tsl(self, idx):
        rand_rot = SO3.exp(self.rand_transf[idx]["rr"]).as_matrix()
        rand_tsl = self.rand_transf[idx]["rt"]
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
        rand_transf_wrt_cam = super().get_obj_transf_wrt_cam(idx).dot(rand_transf)
        return rand_transf_wrt_cam.astype(np.float32)

    def get_obj_verts_transf(self, idx):
        obj_can = self.get_obj_verts_can(idx)[0]
        obj_can = np.concatenate([obj_can, np.ones((obj_can.shape[0], 1))], axis=1)
        obj_transf = self.get_obj_transf_wrt_cam(idx).dot(obj_can.T).T[:, :3]
        return obj_transf.astype(np.float32)

    def get_obj_corners3d(self, idx):
        obj_corners = super().get_obj_corners3d(idx)
        obj_transf = super().get_obj_transf_wrt_cam(idx)
        obj_rot = obj_transf[:3, :3]  # (3, 3)
        obj_tsl = obj_transf[:3, 3:]  # (3, 1)
        rand_rot, rand_tsl = self._get_rand_rot_tsl(idx)
        obj_corners = obj_corners - obj_tsl.T
        obj_corners = np.linalg.inv(obj_rot).dot(obj_corners.T).T
        rand_hand = rand_rot.dot(obj_corners.T).T + rand_tsl
        rand_hand = (obj_rot.dot(rand_hand.T) + obj_tsl).T
        # rand_hand = np.array(rand_hand).astype(np.float32)
        return rand_hand.astype(np.float32)


def main(args):
    import cv2

    ho_dataset = HOdata.get_dataset(
        dataset="fhbsynt",
        data_root="data",
        data_split=args.data_split,
        split_mode=args.split_mode,
        use_cache=True,
        mini_factor=1,
        center_idx=9,
        enable_contact=True,
        like_v1=True,
        filter_no_contact=True,
        filter_thresh=10,
        block_rot=True,
        synt_factor=args.synt_factor,
    )

    from hocontact.utils.logger import logger
    import prettytable as pt

    idx = np.random.randint(len(ho_dataset))
    # contact_info = ho_dataset.get_processed_contact_info(idx)
    # print(f"CONTACT_INFO KEYS : {[key for key in contact_info.keys()]}", "yellow")

    sample = ho_dataset[idx]
    tb = pt.PrettyTable(padding_width=3, header=False)
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            tb.add_row([key, type(value), value.shape])
        elif isinstance(value, torch.Tensor):
            tb.add_row([key, type(value), tuple(value.size())])
        else:
            tb.add_row([key, type(value), value])
    print(f"{'='*40} ALL FHB SAMPLE KEYS {'>'*40}", "blue")
    print(str(tb))

    if args.vis:

        def view_data(ho_dataset):
            for i in range(len(ho_dataset)):

                joint2d = ho_dataset.get_joints2d(i)

                # TEST: obj_transf @ obj_verts_can == obj_verts_transf >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                obj_transf = ho_dataset.get_obj_transf_wrt_cam(i)
                obj_rot = obj_transf[:3, :3]  # (3, 3)
                obj_tsl = obj_transf[:3, 3:]  # (3, 1)

                obj_verts_can, _, __ = ho_dataset.get_obj_verts_can(i)  # (N, 3)
                obj_verts_pred = (obj_rot.dot(obj_verts_can.transpose()) + obj_tsl).transpose()
                obj_verts2d_pred = ho_dataset.project(obj_verts_pred, ho_dataset.cam_intr)

                obj_verts2d_gt = ho_dataset.get_obj_verts2d(i)

                # # TEST: obj_transf @ obj_corners_can == obj_corners_3d >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # obj_corners_can = ho_dataset.get_obj_corners_can(i)
                # obj_corners_transf = (obj_rot.dot(obj_corners_can.transpose()) + obj_tsl).transpose()
                # obj_corners2d = ho_dataset.project(obj_corners_transf, ho_dataset.cam_intr)

                # TEST: MANO(get_hand_pose_wrt_cam) + get_hand_tsl_wrt_cam == get_hand_verts3d >>>>>>>>>>>>>>>>>>>>>>>>>>
                hand_pose = torch.from_numpy(ho_dataset.get_hand_pose_wrt_cam(i)).unsqueeze(0)
                hand_shape = torch.from_numpy(ho_dataset.get_hand_shape(i)).unsqueeze(0)
                hand_tsl = ho_dataset.get_hand_tsl_wrt_cam(i)

                hand_verts, hand_joints = ho_dataset.layer(hand_pose, hand_shape)
                hand_verts = np.array(hand_verts.squeeze(0)) + hand_tsl
                hand_verts_2d = ho_dataset.project(hand_verts, ho_dataset.cam_intr)
                hand_verts_2dgt = ho_dataset.get_hand_verts2d(i)

                img = ho_dataset.get_image(i)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for j in range(obj_verts2d_pred.shape[0]):
                    v = obj_verts2d_pred[j]
                    cv2.circle(img, (v[0], v[1]), radius=2, thickness=-1, color=(0, 255, 0))

                # for j in range(obj_corners2d.shape[0]):
                #     v = obj_corners2d[j]
                #     cv2.circle(img, (v[0], v[1]), radius=1, thickness=-1, color=(255, 255, 0))

                # for j in range(hand_verts_2d.shape[0]):
                #     v = hand_verts_2d[j]
                #     cv2.circle(img, (v[0], v[1]), radius=3, thickness=-1, color=(255, 0, 0))

                for j in range(hand_verts_2dgt.shape[0]):
                    v = hand_verts_2dgt[j]
                    cv2.circle(img, (v[0], v[1]), radius=1, thickness=-1, color=(255, 255, 0))
                cv2.imshow("fhbhands", img)
                cv2.waitKey(0)

        view_data(ho_dataset)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="test fhbdataset")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--data_split", choices=["train", "test", "all"], default="train", type=str)
    parser.add_argument("--split_mode", choices=["subjects", "actions"], default="actions", type=str)
    parser.add_argument("--synt_factor", type=int, choices=[1, 2, 3], default=1)
    main(parser.parse_args())
