import numpy as np
import torch
from .ho3dsynt import HO3DSynt
from .hodata import HOdata
from models.CPF.utils.visualize.vis_contact_info import view_vertex_contact
from liegroups import SE3, SO3
from manopth import manolayer
from scipy.optimize._lsq.least_squares import least_squares

class HO3DGen(HO3DSynt):
    def __init__(
        self,
        data_root="data",
        data_split="train",
        njoints=21,
        use_cache=True,
        enable_contact=False,
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
        synt_factor=12,
        # *<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        **kwargs,
    ):
        super().__init__(
            data_root=data_root,
            data_split=data_split,
            njoints=njoints,
            use_cache=use_cache,
            enable_contact=enable_contact,
            filter_no_contact=False,  # ho3dgen don't need to check
            filter_thresh=filter_thresh,
            mini_factor=1.0,  # TODO: ho3dgen does not support mini_factor
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
        self.name = "HO3D_gen"
        self.rand_size = synt_factor

    def load_dataset(self):
        assert self.split_mode == "objects", "ho3dgen only supports objects split"
        super().load_dataset()
        self.can_transf = []
        self.hand_fitting_pose = []
        for i in range(len(self)):
            self.can_transf.append(self._get_obj_can_transf(i))
            self.hand_fitting_pose.append(self._get_hand_fitting_pose(i))

    def load_contact_infos(self, idxs, seq_map):
        contact_info = []
        for i in range(len(idxs)):
            seq, idx = idxs[i]
            path = seq_map[seq][idx]["img"]
            path = path.replace(self.root, self.root_supp)
            path = path.replace("rgb", "contact_info")
            path = path.replace("png", "pkl")
            contact_info.append(path)
        return contact_info

    def _get_obj_can_transf(self, idx):
        path = self.get_image_path(idx)
        path = path.replace("rgb", "meta")
        path = path.replace(".png", ".npy")

        transf = np.load(path)
        return transf.astype(np.float32)

    def _get_hand_fitting_pose(self, idx):
        path = self.get_image_path(idx)
        path = path.replace("rgb", "pose")
        path = path.replace(".png", ".npy")

        pose = np.load(path)
        return pose.astype(np.float32)

    def get_obj_transf_wrt_cam(self, idx):
        return super().get_obj_transf_wrt_cam(idx) @ self.can_transf[idx]

    # def get_hand_pose_wrt_cam(self, idx):
    #     annot = self.get_annot(idx)
    #     return annot["hp"].astype(np.float32)
    def get_hand_pose_wrt_cam(self, idx):
        return self.hand_fitting_pose[idx]

    # def get_hand_tsl_wrt_cam(self, idx):
    #     annot = self.get_annot(idx)
    #     return annot["ht"].astype(np.float32)


def fit_transf(raw_joints, target_joints):
    raw_joints = np.concatenate([raw_joints, np.ones((raw_joints.shape[0], 1), dtype=np.float32)], axis=1)

    def ls_loss(se3):
        transf = SE3.exp(se3).as_matrix()
        joints = (transf @ raw_joints.T).T[:, :3]
        res = joints.ravel() - target_joints.ravel()
        return res

    return least_squares(ls_loss, np.ones(6, dtype=np.float32)).x


def cal_transf(raw_pose, hand_verts, hand_shape):
    """
    input: raw_pose (45, ) hand_verts [778, 3], hand_shape (10, )
    """
    rot = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, requires_grad=True)
    tsl = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, requires_grad=True)

    th_betas = torch.from_numpy(hand_shape).float().unsqueeze(0)
    hand_verts = torch.from_numpy(hand_verts).float()
    raw_pose = torch.from_numpy(raw_pose).float()

    optimizer = torch.optim.Adam([rot, tsl], lr=1e-1)

    if not hasattr(cal_transf, "mano_l"):
        cal_transf.mano_l = manolayer.ManoLayer(
            joint_rot_mode="axisang", use_pca=False, mano_root="assets/mano", center_idx=None, flat_hand_mean=True,
        )

    for idx in range(1000):
        hand_pose = torch.cat((rot, raw_pose), dim=0).unsqueeze(0)
        new_verts, _ = cal_transf.mano_l(hand_pose, th_betas=th_betas, th_trans=tsl.unsqueeze(0))
        new_verts = new_verts.squeeze(0)
        loss = torch.mean(torch.pow((new_verts.flatten() - hand_verts.flatten()), 2)) * 1000.0
        optimizer.zero_grad()
        if idx > 300 and loss.item() < 1e-9:
            break
        loss.backward()
        optimizer.step()

    return rot.detach().numpy(), tsl.detach().numpy()


def main(args):
    import cv2

    ho_dataset = HOdata.get_dataset(
        dataset="ho3dgen",
        data_root="data",
        data_split=args.data_split,
        split_mode=args.split_mode,
        use_cache=False,
        mini_factor=1,
        center_idx=9,
        enable_contact=True,
        like_v1=True,
        filter_no_contact=False,
        filter_thresh=10.0,
        block_rot=True,
        synt_factor=12,
    )
    # raw_dataset = HOdata.get_dataset(
    #     dataset="ho3d",
    #     data_root="data",
    #     data_split=args.data_split,
    #     split_mode=args.split_mode,
    #     use_cache=True,
    #     mini_factor=1.0,
    #     center_idx=9,
    #     enable_contact=True,
    #     like_v1=True,
    #     filter_no_contact=False,
    #     filter_thresh=10.0,
    #     block_rot=True,
    # )
    # for i in range(len(raw_dataset)):
    #     if "MC6" in raw_dataset.get_image_path(i):
    #         for j in range(12):
    #             assert np.allclose(
    #                 raw_dataset.get_obj_transf_wrt_cam(i), ho_dataset.get_hand_pose_wrt_cam((i - 2488) * 12 + j)
    #             )
    # print(raw_dataset.get_obj_transf_wrt_cam(2488))
    # print(ho_dataset.get_hand_pose_wrt_cam(0, raw_dataset))
    # assert False
    # new_rot = []
    # new_tsl = []
    import pickle

    import prettytable as pt

    # hand_poses = []
    # hand_tsls = []
    # cam_extr = raw_dataset.cam_extr[:3, :3]
    # for i in tqdm(range(len(ho_dataset))):
    #     raw_joints = (cam_extr @ raw_dataset.get_joints3d(i // 12 + 2488).T).T
    #     assert np.allclose(raw_dataset.get_joints3d(i // 12 + 2488), HO3D.get_joints3d(ho_dataset, i))
    #     target_joints = (cam_extr @ ho_dataset.get_joints3d(i).T).T
    #     res = fit_transf(raw_joints, target_joints)
    #     raw_annot = raw_dataset.get_annot(i // 12 + 2488)
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
    #     with open(ho_dataset.get_image_path(i).replace("rgb", "pose").replace("png", "npy"), "wb") as f:
    #         np.save(f, hand_pose.astype(np.float32))
    #     # if i > 10:
    #     #     break
    # assert False
    # hand_poses = []
    # hand_tsls = []
    # for i in tqdm(range(len(ho_dataset))):
    #     raw_h_shape = raw_dataset.get_hand_shape(i // 12 + 2488)
    #     tar_h_verts = ho_dataset.get_hand_verts3d(i)
    #     raw_hand_pose = raw_dataset.get_hand_pose_wrt_cam(i // 12 + 2488)
    #     root_rot, tsl = cal_transf(raw_hand_pose[3:], tar_h_verts, raw_h_shape)
    #     hand_pose = np.concatenate([root_rot, raw_hand_pose[3:]], axis=0)
    #     hand_tsl = tsl
    #     with open(ho_dataset.get_image_path(i).replace("rgb", "meta").replace("png", "pkl"), "rb") as f:
    #         pkl = pickle.load(f)
    #     pkl["hp"] = hand_pose
    #     pkl["ht"] = hand_tsl
    #     hand_poses.append(hand_pose.astype(np.float32))
    #     hand_tsls.append(hand_tsl.astype(np.float32))
    #     with open(ho_dataset.get_image_path(i).replace("rgb", "meta").replace("png", "pkl"), "wb") as f:
    #         pickle.dump(pkl, f)
    # assert False
    from hocontact.utils.logger import logger
    from tqdm import tqdm

    # idx = np.random.randint(len(ho_dataset))
    # print(ho_dataset.get_image_path(idx))
    # sample = ho_dataset[idx]
    # tb = pt.PrettyTable(padding_width=3, header=False)
    # for key, value in sample.items():
    #     if isinstance(value, np.ndarray):
    #         tb.add_row([key, type(value), value.shape])
    #     elif isinstance(value, torch.Tensor):
    #         tb.add_row([key, type(value), tuple(value.size())])
    #     else:
    #         tb.add_row([key, type(value), value])
    # print(f"{'=' * 40} ALL HO3D SAMPLE KEYS {'>' * 40}", "blue")
    # print(str(tb))
    if args.render:
        view_vertex_contact(ho_dataset)

    if args.vis:

        def view_data(ho_dataset):
            for i in range(len(ho_dataset)):
                # i = np.random.randint(len(ho_dataset))
                objverts2d = ho_dataset.get_obj_verts2d(i)
                joint2d = ho_dataset.get_joints2d(i)

                # TEST: obj_transf @ obj_verts_can == obj_verts_transf >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                obj_transf = ho_dataset.get_obj_transf_wrt_cam(i)
                obj_rot = obj_transf[:3, :3]  # (3, 3)
                obj_tsl = obj_transf[:3, 3:]  # (3, 1)

                obj_verts_can, _, __ = ho_dataset.get_obj_verts_can(i)  # (N, 3)
                obj_verts_pred = (obj_rot.dot(obj_verts_can.transpose()) + obj_tsl).transpose()
                assert np.allclose(ho_dataset.get_obj_verts_transf(i), obj_verts_pred)
                obj_verts2d_pred = ho_dataset.project(obj_verts_pred, ho_dataset.get_cam_intr(i))

                # TEST: obj_transf @ obj_corners_can == obj_corners_3d >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                obj_corners_can = ho_dataset.get_obj_corners_can(i)
                obj_corners_transf = (obj_rot.dot(obj_corners_can.transpose()) + obj_tsl).transpose()
                obj_corners2d = ho_dataset.project(obj_corners_transf, ho_dataset.get_cam_intr(i))

                # TEST: MANO(get_hand_pose_wrt_cam) + get_hand_tsl_wrt_cam == get_hand_verts3d >>>>>>>>>>>>>>>>>>>>>>>>>>
                hand_pose = torch.from_numpy(ho_dataset.get_hand_pose_wrt_cam(i)).unsqueeze(0)
                hand_shape = torch.from_numpy(ho_dataset.get_hand_shape(i)).unsqueeze(0)
                hand_tsl = ho_dataset.get_hand_tsl_wrt_cam(i)
                # hand_pose = torch.from_numpy(hand_poses[i]).unsqueeze(0)
                # hand_tsl = hand_tsls[i]

                hand_verts, _ = ho_dataset.layer(hand_pose, hand_shape)
                hand_verts = np.array(hand_verts.squeeze(0)) + hand_tsl
                # hand_verts = (cam_extr @ hand_verts.T).T
                # print(hand_verts - ho_dataset.get_hand_verts3d(i))
                hand_verts_2d = ho_dataset.project(hand_verts, ho_dataset.get_cam_intr(i))
                hand_joints_2dgt = ho_dataset.get_joints2d(i)
                hand_verts_2dgt = ho_dataset.get_hand_verts2d(i)
                img = ho_dataset.get_image(i)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for j in range(obj_verts2d_pred.shape[0]):
                    v = obj_verts2d_pred[j]
                    cv2.circle(img, (v[0], v[1]), radius=1, thickness=-1, color=(0, 255, 0))
                for j in range(hand_joints_2dgt.shape[0]):
                    v = hand_joints_2dgt[j]
                    cv2.circle(img, (v[0], v[1]), radius=8, thickness=-1, color=(255, 255, 255))
                for j in range(hand_verts_2dgt.shape[0]):
                    v = hand_verts_2dgt[j]
                    cv2.circle(img, (v[0], v[1]), radius=3, thickness=-1, color=(255, 255, 0))
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
