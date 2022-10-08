import os
import pickle
from ast import parse
from os.path import join

import cv2
from . import ho3d
from . import hodata
import numpy as np
import torch
import trimesh
from . import ho3dutils
from .fhb import main
from .hodata import HOdata
from .hoquery import BaseQueries, ContactQueries, MetaQueries, get_trans_queries
from ..utils import meshutils
from ..utils.visualize.vis_contact_info import view_vertex_contact
from liegroups import SO3
from manopth import manolayer
from manopth.anchorutils import anchor_load_driver
from PIL import Image
from termcolor import cprint
from torch import dtype, float32, select


class HO3DYCBA(ho3d.HO3D):
    def __init__(
        self,
        data_root="data",
        data_split="train",
        njoints=21,
        use_cache=True,
        enable_contact=False,
        filter_no_contact=True,
        filter_thresh=10,
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
            mini_factor=1.0,  # ho3dycba does not support mini_factor
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

        self.name = "HO3D_ycba"
        self.layer = manolayer.ManoLayer(
            mano_root="assets/mano/", side="right", use_pca=True, ncomps=45, flat_hand_mean=True
        )
        self.cam_intr = np.array([[617.343, 0.0, 312.42], [0.0, 617.343, 241.42], [0.0, 0.0, 1.0]]).astype(np.float32)

    def load_dataset(self):
        self._preload()

        cache_folder = os.path.dirname(self.cache_path)
        os.makedirs(cache_folder, exist_ok=True)
        self.hand_palm_vertex_index = np.loadtxt(os.path.join(self.root_extra_info, "hand_palm_full.txt"), dtype=np.int)

        if self.split_mode == "objects":
            seqs, subfolder = ho3dutils.get_object_seqs(self.data_split, self.like_v1, self.name)
            print(f"{self.name} {self.data_split} set has sequence {seqs}", "yellow")
            seq_frames, subfolder = self.load_seq_frames(subfolder, seqs)
        elif self.split_mode == "paper":
            seq_frames, subfolder = self.load_seq_frames()
            print(f"{self.name} {self.data_split} set has frames {len(seq_frames)}", "yellow")
        elif self.split_mode == "official":
            seq_frames, subfolder = ho3dutils.get_offi_frames(self.name, self.data_split, self.root)
            print(f"{self.name} {self.data_split} set has frames {len(seq_frames)}", "yellow")
        else:
            assert False

        if os.path.exists(self.cache_path) and self.use_cache:
            with open(self.cache_path, "rb") as p_f:
                annotations = pickle.load(p_f)
            print(f"Loaded cache information for dataset {self.name} from {self.cache_path}")
        else:
            seq = set()
            for sf in seq_frames:
                seq.add(sf[0])
            obj_names = ho3dutils.get_seq_object(list(seq))
            print(f"{self.name} {self.data_split} set use objects: {obj_names}", "yellow")
            annotations = {}
            annotations["img_path"] = []
            annotations["transf"] = []
            annotations["hand_verts"] = []
            annotations["obj_raw_mesh"] = {}
            annotations["obj_corner"] = {}
            annotations["hand_pose"] = []
            annotations["hand_tsl"] = []
            annotations["hand_shape"] = []
            if self.enable_contact:
                annotations["contact_info"] = []
            for name in obj_names:
                img_path = os.path.join(self.data_root, self.name, name, "rgb")
                obj_raw_path = os.path.join(self.data_root, "YCB_models_supp", name, "textured_simple_ds.obj")
                raw_mesh = trimesh.load(obj_raw_path)
                annotations["obj_raw_mesh"][name] = (
                    np.asarray(raw_mesh.vertices, dtype=np.float32),
                    np.asarray(raw_mesh.faces, dtype=np.long),
                )
                annotations["obj_corner"][name] = np.load(
                    os.path.join(self.data_root, "YCB_models_supp", f"{name}/{name}.npy")
                )
                for sample in os.listdir(img_path):
                    path = os.path.join(img_path, sample)
                    meta_path = path.replace("rgb", "meta").replace("png", "pkl")
                    meta = pickle.load(open(meta_path, "rb"))
                    # if (
                    #     self.filter_no_contact
                    #     and self.min_contact_dis(meta, annotations["obj_raw_mesh"][name]) > self.filter_thresh
                    # ):
                    #     continue
                    annotations["img_path"].append(path)
                    if self.enable_contact:
                        annotations["contact_info"].append(
                            path.replace("rgb", "contact_info")
                            .replace("png", "pkl")
                            .replace(self.name, f"{self.name}_supp")
                        )
                    annotations["transf"].append(meta["transf"])
                    annotations["hand_verts"].append(meta["hand_v"])
                    annotations["hand_pose"].append(meta["hand_p"])
                    annotations["hand_tsl"].append(meta["hand_t"])
                    annotations["hand_shape"].append(meta["hand_s"])
            with open(self.cache_path, "wb") as p_f:
                pickle.dump(annotations, p_f)
            print("Wrote cache for dataset {} to {}".format(self.name, self.cache_path), "yellow")
        self.img_path = annotations["img_path"]
        self.transf = annotations["transf"]
        self.hand_verts = annotations["hand_verts"]
        self.raw_meshes = annotations["obj_raw_mesh"]
        self.obj_corners = annotations["obj_corner"]
        self.hand_pose = annotations["hand_pose"]
        self.hand_tsl = annotations["hand_tsl"]
        self.hand_shape = annotations["hand_shape"]
        if self.enable_contact:
            self.contact_infos = annotations["contact_info"]
        self.n_palm_vert = self.get_n_hand_palm_vert(0)
        if self.enable_contact:
            (
                self.anchor_face_vertex_index,
                self.anchor_weights,
                self.hand_vertex_merged_assignment,
                self.anchor_mapping,
            ) = anchor_load_driver(self.root_extra_info)

        print(f"{self.name} Got {len(self)} samples for data_split {self.data_split}")
        print(f"Got {len(self)} samples for data_split {self.data_split}")

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

    def __len__(self):
        return len(self.img_path)

    def get_obj_name(self, idx):
        return self.img_path[idx].split("/")[-3]

    def get_obj_verts_can(self, idx):
        verts = self.raw_meshes[self.get_obj_name(idx)][0]
        verts = self.cam_extr[:3, :3].dot(verts.transpose()).transpose()
        # NOTE: verts_can = verts - bbox_center
        verts_can, bbox_center, bbox_scale = meshutils.center_vert_bbox(verts, scale=False)  # !! CENTERED HERE
        return verts_can, bbox_center, bbox_scale

    def get_obj_faces(self, idx):
        return self.raw_meshes[self.get_obj_name(idx)][1].astype(np.int32)

    def get_obj_transf_wrt_cam(self, idx):
        return self.transf[idx]

    def get_obj_verts_transf(self, idx):
        verts = self.get_obj_verts_can(idx)[0]
        verts = np.concatenate([verts, np.ones((verts.shape[0], 1), dtype=np.float32)], axis=1)
        verts = (self.get_obj_transf_wrt_cam(idx) @ verts.T).T[:, :3]
        return verts.astype(np.float32)

    def get_hand_verts3d(self, idx):
        return self.hand_verts[idx].astype(np.float32)

    def get_obj_corners_can(self, idx):
        corners = self.obj_corners[self.get_obj_name(idx)]
        corners = self.cam_extr[:3, :3].dot(corners.transpose()).transpose()
        return corners.astype(np.float32)

    def get_image(self, idx):
        return Image.open(self.img_path[idx]).convert("RGB")

    def get_image_path(self, idx):
        return self.img_path[idx]

    def get_cam_intr(self, idx):
        return self.cam_intr

    def get_joints3d(self, idx):
        hand_pose = torch.from_numpy(self.hand_pose[idx]).float().unsqueeze(0)
        hand_shape = torch.from_numpy(self.hand_shape[idx]).float().unsqueeze(0)
        hand_tsl = torch.from_numpy(self.hand_tsl[idx]).float().unsqueeze(0)
        _, joints = self.layer(hand_pose, th_betas=hand_shape, th_trans=hand_tsl)
        return joints.squeeze(0).numpy().astype(np.float32)

    def get_hand_pose_wrt_cam(self, idx):
        return self.hand_pose[idx].astype(np.float32)

    def get_hand_tsl_wrt_cam(self, idx):
        return self.hand_tsl[idx].astype(np.float32)

    def get_hand_shape(self, idx):
        return self.hand_shape[idx].astype(np.float32)

    def get_sample_identifier(self, idx):
        identifier = (
            f"{self.data_split}_{self.split_mode}_mf{self.mini_factor}"
            f"_likev1{''if self.like_v1 else '(x)'}"
            f"_fct{self.filter_thresh if self.filter_no_contact else '(x)'}"
            f"_ec{'' if self.enable_contact else '(x)'}"
        )

        res = f"{self.name}/{identifier}/{idx}"
        return res


def main(args):
    import cv2

    ho_dataset = HOdata.get_dataset(
        dataset="ho3dycba",
        data_root="data",
        data_split="test",
        split_mode="official",
        use_cache=False,
        mini_factor=1,
        center_idx=9,
        enable_contact=True,
        like_v1=True,
        filter_no_contact=False,
        filter_thresh=10.0,
        block_rot=True,
        synt_factor=1,
    )

    import prettytable as pt
    from hocontact.utils.logger import logger

    idx = np.random.randint(len(ho_dataset))

    sample = ho_dataset[idx]
    tb = pt.PrettyTable(padding_width=3, header=False)
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            tb.add_row([key, type(value), value.shape])
        elif isinstance(value, torch.Tensor):
            tb.add_row([key, type(value), tuple(value.size())])
        else:
            tb.add_row([key, type(value), value])
    print(f"{'=' * 40} ALL HO3D SAMPLE KEYS {'>' * 40}", "blue")
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

    parser = argparse.ArgumentParser(description="test ho3d dataset")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--data_split", choices=["train", "test", "all"], default="train", type=str)
    parser.add_argument("--split_mode", choices=["", "objects"], default="objects", type=str)
    main(parser.parse_args())
