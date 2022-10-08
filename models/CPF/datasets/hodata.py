from abc import ABC, abstractmethod
import numpy as np
from PIL import Image, ImageFilter
from manopth import manolayer
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader, Dataset
from ..utils import func
from .hoquery import (
    BaseQueries,
    ContactQueries,
    TransQueries,
    MetaQueries,
    get_trans_queries,
    one_query_in,
    match_collate_queries,
    recover_queries,
    CollateQueries
)
from ..utils import augutils
from ..utils import handutils
from ..utils.eval import Evaluator
from torch.utils.data.dataset import ConcatDataset

from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.pipelines import Compose
from mmcv.parallel.data_container import DataContainer

def ho_collate(batch):
    return hodata_collate(batch)


def hodata_collate(batch):
    """
    Collate function, duplicating the items in extend_queries along the
    first dimension so that they all have the same length.
    Typically applies to faces and vertices, which have different sizes
    depending on the object.
    """
    # *  NEW QUERY: CollateQueries.PADDING_MASK

    extend_queries = {
        TransQueries.OBJ_VERTS_3D,
        BaseQueries.OBJ_VERTS_3D,
        BaseQueries.OBJ_CAN_VERTS,
        BaseQueries.OBJ_VERTS_2D,
        BaseQueries.OBJ_VIS_2D,
        TransQueries.OBJ_VERTS_2D,
        BaseQueries.OBJ_FACES,
        ContactQueries.VERTEX_CONTACT,
        ContactQueries.CONTACT_REGION_ID,
        ContactQueries.CONTACT_ANCHOR_ID,
        ContactQueries.CONTACT_ANCHOR_DIST,
        ContactQueries.CONTACT_ANCHOR_ELASTI,
        ContactQueries.CONTACT_ANCHOR_PADDING_MASK,
    }

    pop_queries = []
    for poppable_query in extend_queries:
        if poppable_query in batch[0]:
            pop_queries.append(poppable_query)

    # Remove fields that don't have matching sizes
    for pop_query in pop_queries:
        padding_query_field = match_collate_queries(pop_query)
        max_size = max([sample[pop_query].shape[0] for sample in batch])
        for sample in batch:
            pop_value = sample[pop_query]
            orig_len = pop_value.shape[0]
            # Repeat vertices so all have the same number
            pop_value = np.concatenate([pop_value] * int(max_size / pop_value.shape[0] + 1))[:max_size]
            sample[pop_query] = pop_value
            if padding_query_field not in sample:
                # !===== this is only done for verts / faces >>>>>
                # generate a new field, contains padding mask
                # note that only the beginning pop_value.shape[0] points are in effect
                # so the mask will be a vector of length max_size, with origin_len ones in the beginning
                padding_mask = np.zeros(max_size, dtype=np.int)
                padding_mask[:orig_len] = 1
                sample[padding_query_field] = padding_mask

    # store the mask filtering the points

    batch = default_collate(batch)
    return batch


def evaluate_collate(batch):
    batch[0].pop('img_metas')
    return default_collate(batch)


class HOdata(Dataset):
    def __init__(
        self,
        data_root="data",
        data_split="train",
        njoints=21,
        use_cache=True,
        filter_no_contact=False,
        filter_thresh=10.0,
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
        assets_root='assets',
    ):
        super().__init__()
        self.data_root = data_root
        self.data_split = data_split
        self.assets_root = assets_root
        self.njoints = njoints
        self.use_cache = use_cache
        self.filter_no_contact = filter_no_contact
        self.filter_thresh = float(filter_thresh)
        self.mini_factor = float(mini_factor)

        self.center_idx = center_idx
        self.scale_jittering = scale_jittering
        self.center_jittering = center_jittering
        self.block_rot = block_rot
        self.max_rot = max_rot
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.blur_radius = blur_radius

        self.sides = sides

        self.layer = manolayer.ManoLayer(
            joint_rot_mode="axisang", use_pca=False,
            mano_root=f'{self.assets_root}/mano',
            center_idx=None, flat_hand_mean=True,
        )

        self.inp_res = (255, 255)  # this will be overried by its subclass

        # get paired links as neighboured joints
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]

        # get queries
        self.all_queries = {
            BaseQueries.IMAGE,
            BaseQueries.JOINTS_2D,
            BaseQueries.JOINTS_3D,
            BaseQueries.OBJ_VERTS_2D,
            BaseQueries.OBJ_VIS_2D,
            BaseQueries.OBJ_VERTS_3D,
            BaseQueries.HAND_VERTS_2D,
            BaseQueries.HAND_VERTS_3D,
            BaseQueries.OBJ_CAN_VERTS,
            BaseQueries.SIDE,
            BaseQueries.CAM_INTR,
            BaseQueries.JOINT_VIS,
            BaseQueries.OBJ_TRANSF,
            BaseQueries.HAND_POSE_WRT_CAM,
            BaseQueries.IMAGE_PATH,
            MetaQueries.SAMPLE_IDENTIFIER,
        }
        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.update(trans_queries)

        self.queries = query
        if query is None:
            self.queries = self.all_queries
        print(f"HOdata(ABC) queries: \n {self.queries}", 'cyan')

    @classmethod
    def get_dataset(
        cls,
        dataset,
        data_root,
        data_split,
        split_mode,
        use_cache,
        mini_factor,
        center_idx,
        enable_contact,
        filter_no_contact,
        filter_thresh,
        synt_factor,
        **kwargs,
    ):

        if dataset in ["fhb", "fhbhand", "fhbhands"]:
            from . import fhb

            synt_factor = 0  # rewrite synt_factor
            fhb = fhb.FPHB(
                data_root=data_root,
                data_split=data_split,
                split_mode=split_mode,
                use_cache=use_cache,
                mini_factor=mini_factor,
                center_idx=center_idx,
                enable_contact=enable_contact,
                filter_no_contact=filter_no_contact,
                filter_thresh=filter_thresh,
                **kwargs,
            )
            fhb.load_dataset()
            return fhb
        elif dataset == "ho3d":
            from .ho3d import HO3D

            synt_factor == 0  # rewrite synt_factor
            ho3d = HO3D(
                data_root=data_root,
                data_split=data_split,
                split_mode=split_mode,
                use_cache=use_cache,
                mini_factor=mini_factor,
                center_idx=center_idx,
                enable_contact=enable_contact,
                filter_no_contact=filter_no_contact,
                filter_thresh=filter_thresh,
                **kwargs,
            )
            ho3d.load_dataset()
            return ho3d
        elif dataset == "ho3dsynt":
            from . import ho3dsynt

            assert synt_factor in [1, 2, 3], "wrong synt factor"
            ho3dsynt = ho3dsynt.HO3DSynt(
                data_root=data_root,
                data_split=data_split,
                split_mode=split_mode,
                use_cache=use_cache,
                mini_factor=mini_factor,
                center_idx=center_idx,
                enable_contact=enable_contact,
                synt_factor=synt_factor,
                filter_no_contact=filter_no_contact,
                filter_thresh=filter_thresh,
                **kwargs,
            )
            ho3dsynt.load_dataset()
            return ho3dsynt
        elif dataset == "ho3dsynttsl":
            from . import ho3dsynttsl

            assert synt_factor in [1, 2, 3, 4, 5], "wrong synt factor"
            ho3dsynttsl = ho3dsynttsl.HO3DSyntTsl(
                data_root=data_root,
                data_split=data_split,
                split_mode=split_mode,
                use_cache=use_cache,
                mini_factor=mini_factor,
                center_idx=center_idx,
                enable_contact=enable_contact,
                synt_factor=synt_factor,
                filter_no_contact=filter_no_contact,
                filter_thresh=filter_thresh,
                **kwargs,
            )
            ho3dsynttsl.load_dataset()
            return ho3dsynttsl
        elif dataset == "ho3dtrick":
            from . import ho3dtrick

            assert synt_factor in [1, 2, 3, 4, 5, 6, 7, 8], "wrong synt factor"
            ho3dtrick = ho3dtrick.HO3DSyntTrick(
                data_root=data_root,
                data_split=data_split,
                split_mode=split_mode,
                use_cache=use_cache,
                mini_factor=mini_factor,
                center_idx=center_idx,
                enable_contact=enable_contact,
                synt_factor=synt_factor,
                filter_no_contact=filter_no_contact,
                filter_thresh=filter_thresh,
                **kwargs,
            )
            ho3dtrick.load_dataset()
            return ho3dtrick
        elif dataset == "ho3dgen":
            from . import ho3dgen

            assert synt_factor in list(range(1, 13)), "wrong synt factor"
            ho3dgen = ho3dgen.HO3DGen(
                data_root=data_root,
                data_split=data_split,
                split_mode=split_mode,
                use_cache=use_cache,
                mini_factor=mini_factor,
                center_idx=center_idx,
                enable_contact=enable_contact,
                synt_factor=synt_factor,  # ? need to be complete
                filter_no_contact=filter_no_contact,
                filter_thresh=filter_thresh,
                **kwargs,
            )
            ho3dgen.load_dataset()
            return ho3dgen
        elif dataset == "ho3dycba":
            from . import ho3dycba

            ho3dycba = ho3dycba.HO3DYCBA(
                data_root=data_root,
                data_split=data_split,
                split_mode=split_mode,
                use_cache=use_cache,
                mini_factor=mini_factor,
                center_idx=center_idx,
                enable_contact=enable_contact,
                **kwargs,
            )
            ho3dycba.load_dataset()
            return ho3dycba
        elif dataset == "fhbsynt":
            from . import fhbsynt

            assert synt_factor in [1, 2, 3], "wrong synt factor"
            fhbsynt = fhbsynt.FPHBSynt(
                data_root=data_root,
                data_split=data_split,
                split_mode=split_mode,
                use_cache=use_cache,
                mini_factor=mini_factor,
                center_idx=center_idx,
                enable_contact=enable_contact,
                synt_factor=synt_factor,
                filter_no_contact=filter_no_contact,
                filter_thresh=filter_thresh,
                **kwargs,
            )
            fhbsynt.load_dataset()
            return fhbsynt
        else:
            raise NotImplementedError(f"Unknown dataset {dataset}")

    def get(self, item, idx):
        return getattr(self, "get_" + item, lambda x: None)(idx)

    # ? only used in offline eval
    # ? not abstract method; will raise unimplemeted exception if not overloaded
    def get_hand_tsl_wrt_cam(self, idx):
        raise RuntimeError("method unimplemented")

    # ? only used in offline eval
    # ? not abstract method; will raise unimplemeted exception if not overloaded
    def get_hand_shape(self, idx):
        raise RuntimeError("method unimplemented")

    # ? only used in offline eval
    # ? not abstract method; will raise unimplemeted exception if not overloaded
    # ! warning: this cannot be directly fed into mano_layer to get hand shape. will result in incorrect shape
    # ! warning: one guess for this is that root transform is taken into consideration in LBS algorithm
    def get_hand_pose_wrt_cam(self, idx):
        raise RuntimeError("method unimplemented")

    # ? only used in offline eval
    # ? not abstract method; will raise unimplemeted exception if not overloaded
    def get_hand_axisang_wrt_cam(self, idx):
        raise RuntimeError("method unimplemented")

    # ? only used in offline eval
    # ? not abstract method; will raise unimplemeted exception if not overloaded
    def get_hand_rot_wrt_cam(self, idx):
        raise RuntimeError("method unimplemented")

    # ? only used in offline eval
    # ? not abstract method; will raise unimplemeted exception if not overloaded
    def get_obj_verts_transf_reduced(self, idx):
        raise RuntimeError("method unimplemented")

    # ? only used in offline eval
    # ? not abstract method; will raise unimplemeted exception if not overloaded
    def get_obj_verts_can_reduced(self, idx):
        raise RuntimeError("method unimplemented")

    # ? only used in offline eval
    # ? not abstract method; will raise unimplemeted exception if not overloaded
    def get_obj_faces_reduced(self, idx):
        raise RuntimeError("method unimplemented")

    # ? only used in post process
    # ? not abstract method; will raise unimplemented exception if not overloaded
    def get_obj_voxel_points_can(self, idx):
        raise RuntimeError("method unimplemented")

    # ? only used in post process
    # ? not abstract method; will raise unimplemented exception if not overloaded
    def get_obj_voxel_points_transf(self, idx):
        raise RuntimeError("method unimplemented")

    # ? only used in post process
    # ? not abstract method; will raise unimplemented exception if not overloaded
    def get_obj_voxel_element_volume(self, idx):
        raise RuntimeError("method unimplemented")

    @staticmethod
    def project(points3d, cam_intr):
        hom_2d = np.array(cam_intr).dot(points3d.transpose()).transpose()
        points2d = (hom_2d / (hom_2d[:, 2:] + 1e-6))[:, :2]
        return points2d.astype(np.float32)

    def __getitem__(self, idx):

        query = self.queries
        sample = {}

        center, scale = self.get_center_scale(idx)

        if BaseQueries.JOINT_VIS in query:
            jointvis = self.get_joint_vis(idx)
            sample[BaseQueries.JOINT_VIS] = jointvis

        # Get sides
        if BaseQueries.SIDE in query:
            hand_side = self.get_sides(idx)
            hand_side, flip = handutils.flip_hand_side(self.sides, hand_side)
            sample[BaseQueries.SIDE] = hand_side
        else:
            flip = False

        # Get original image
        if BaseQueries.IMAGE in query or TransQueries.IMAGE in query:
            img = self.get_image(idx)
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if BaseQueries.IMAGE in query:
                sample[BaseQueries.IMAGE] = np.array(img)

        if BaseQueries.IMAGE_PATH in query:
            sample[BaseQueries.IMAGE_PATH] = self.get_image_path(idx)

        # Flip and image 2d if needed
        if flip:
            center[0] = self.inp_res[0] - center[0]

        # Data augmentation
        if self.data_split in ["train", "trainval"]:
            # ! Randomly jitter center
            # Center is located in square of size 2*center_jitter_factor
            # in center of cropped image
            center_jit = Uniform(low=-1, high=1).sample((2,)).numpy()
            center_offsets = self.center_jittering * scale * center_jit
            center = center + center_offsets.astype(int)

            # Scale jittering
            scale_jit = Normal(0, 1).sample().item() + 1
            scale_jittering = self.scale_jittering * scale_jit
            scale_jittering = np.clip(scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
            scale = scale * scale_jittering
            # tycoer
            if self.max_rot != 0:
                rot = Uniform(low=-self.max_rot, high=self.max_rot).sample().item()
            else:
                rot = 0
        else:
            rot = 0
        if self.block_rot:
            rot = 0
        rot_mat = np.array([[np.cos(rot), -np.sin(rot), 0], [np.sin(rot), np.cos(rot), 0], [0, 0, 1]]).astype(np.float32)

        camintr = self.get_cam_intr(idx)
        if BaseQueries.CAM_INTR in query:
            sample[BaseQueries.CAM_INTR] = camintr.astype(np.float32)

        affinetrans, post_rot_trans = augutils.get_affine_transform(
            center=center,
            scale=scale,
            optical_center=[camintr[0, 2], camintr[1, 2]],  # (cx, cy)print(intr[0, 0])
            out_res=self.inp_res,
            rot=rot,
        )

        if TransQueries.CAM_INTR in query:
            # ! Rotation is applied as extr transform
            new_camintr = post_rot_trans.dot(camintr)
            sample[TransQueries.CAM_INTR] = new_camintr.astype(np.float32)

        if TransQueries.AFFINETRANS in query:
            sample[TransQueries.AFFINETRANS] = affinetrans

        # * =============== ALL 2D POINTS ANNOTATION ===================================================================
        # * =============== Get 2D object points
        if (BaseQueries.OBJ_VERTS_2D in query) or (TransQueries.OBJ_VERTS_2D in query):
            objverts2d = self.get_obj_verts2d(idx)
            if flip:
                objverts2d = objverts2d.copy()
                objverts2d[:, 0] = self.inp_res[0] - objverts2d[:, 0]
            if BaseQueries.OBJ_VERTS_2D in query:
                sample[BaseQueries.OBJ_VERTS_2D] = objverts2d.astype(np.float32)
            if TransQueries.OBJ_VERTS_2D in query:
                transobjverts2d = augutils.transform_coords(objverts2d, affinetrans)
                sample[TransQueries.OBJ_VERTS_2D] = np.array(transobjverts2d).astype(np.float32)
            if BaseQueries.OBJ_VIS_2D in query:
                objvis2d = self.get_obj_vis2d(idx)
                sample[BaseQueries.OBJ_VIS_2D] = objvis2d

        # * ============== Get 2D object corner points
        if (BaseQueries.OBJ_CORNERS_2D in query) or (TransQueries.OBJ_CORNERS_2D in query):
            objcorners2d = self.get_obj_corners2d(idx)
            if flip:
                objcorners2d = objcorners2d.copy()
                objcorners2d[:, 0] = self.inp_res[0] - objcorners2d[:, 0]
            if BaseQueries.OBJ_CORNERS_2D in query:
                sample[BaseQueries.OBJ_CORNERS_2D] = np.array(objcorners2d)
            if TransQueries.OBJ_CORNERS_2D in query:
                transobjcorners2d = augutils.transform_coords(objcorners2d, affinetrans)
                sample[TransQueries.OBJ_CORNERS_2D] = np.array(transobjcorners2d)

        # * ============== Get 2D hand joints
        if (BaseQueries.JOINTS_2D in query) or TransQueries.JOINTS_2D in query:
            joints2d = self.get_joints2d(idx)
            if flip:
                joints2d = joints2d.copy()
                joints2d[:, 0] = self.inp_res[0] - joints2d[:, 0]
            if BaseQueries.JOINTS_2D in query:
                sample[BaseQueries.JOINTS_2D] = joints2d.astype(np.float32)
            if TransQueries.JOINTS_2D in query:
                rows = augutils.transform_coords(joints2d, affinetrans)
                sample[TransQueries.JOINTS_2D] = np.array(rows).astype(np.float32)

        # * =============== Get 2d hand joints visibilities
        if BaseQueries.HAND_VIS_2D in query:
            handvis2d = self.get_hand_vis2d(idx)
            sample[BaseQueries.HAND_VIS_2D] = handvis2d

        # * ============== Get 2D hand verts
        if (BaseQueries.HAND_VERTS_2D in query) or (TransQueries.HAND_VERTS_2D in query):
            handverts2d = self.get_hand_verts2d(idx)
            if flip:
                handverts2d = handverts2d.copy()
                handverts2d[:, 0] = self.inp_res[0] - handverts2d[:, 0]
            if BaseQueries.HAND_VERTS_2D in query:
                sample[BaseQueries.HAND_VERTS_2D] = handverts2d
            if TransQueries.HAND_VERTS_2D in query:
                transhandverts2d = augutils.transform_coords(handverts2d, affinetrans)
                sample[TransQueries.HAND_VERTS_2D] = np.array(transhandverts2d)
        # * <<<<<<<<<<<<<< END OF ALL 2D POINTS ANNOTATION<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ? ============== ALL 3D POINTS ANNOTATION ====================================================================
        # ? ============== Get 3D hand joints & 3D hand centers
        if BaseQueries.JOINTS_3D in query or TransQueries.JOINTS_3D in query:
            joints3d = self.get_joints3d(idx)
            if flip:
                joints3d = joints3d.copy()
                joints3d[:, 0] = -joints3d[:, 0]
            if BaseQueries.JOINTS_3D in query:
                sample[BaseQueries.JOINTS_3D] = joints3d.astype(np.float32)
            # Compute 3D center
            if self.center_idx is not None:
                center3d = joints3d[self.center_idx]
            else:
                center3d = (joints3d[9] + joints3d[0]) / 2  # palm
            sample[BaseQueries.CENTER_3D] = center3d  # Always Compute

            if TransQueries.JOINTS_3D in query:
                joints3d = rot_mat.dot(joints3d.transpose(1, 0)).transpose()
                sample[TransQueries.JOINTS_3D] = joints3d.astype(np.float32)
                # Compute 3D center in TransQueries
                if self.center_idx is not None:
                    center3d = joints3d[self.center_idx]
                else:
                    center3d = (joints3d[9] + joints3d[0]) / 2  # palm
                sample[TransQueries.CENTER_3D] = center3d  # Always Compute

        # ? =============== Get 3D hand vertices
        if BaseQueries.HAND_VERTS_3D in query or TransQueries.HAND_VERTS_3D in query:
            hand_verts3d = self.get_hand_verts3d(idx)
            if flip:
                hand_verts3d = hand_verts3d.copy()
                hand_verts3d[:, 0] = -hand_verts3d[:, 0]
            if BaseQueries.HAND_VERTS_3D in query:
                sample[BaseQueries.HAND_VERTS_3D] = hand_verts3d.astype(np.float32)
            if TransQueries.HAND_VERTS_3D in query:
                hand_verts3d = rot_mat.dot(hand_verts3d.transpose(1, 0)).transpose()
                # ! CAUTION! may lead to unexpected bug
                # // if self.center_idx is not None:
                # //     hand_verts3d = hand_verts3d - center3d
                sample[TransQueries.HAND_VERTS_3D] = hand_verts3d.astype(np.float32)

        # ? =============== Get 3D obj vertices
        if BaseQueries.OBJ_VERTS_3D in query or TransQueries.OBJ_VERTS_3D in query:
            obj_verts3d = self.get_obj_verts_transf(idx)
            if flip:
                obj_verts3d = obj_verts3d.copy()
                obj_verts3d[:, 0] = -obj_verts3d[:, 0]
            if BaseQueries.OBJ_VERTS_3D in query:
                sample[BaseQueries.OBJ_VERTS_3D] = obj_verts3d
            if TransQueries.OBJ_VERTS_3D in query:
                obj_verts3d = rot_mat.dot(obj_verts3d.transpose(1, 0)).transpose()
                # ! CAUTION! may lead to unexpected bug
                # // if self.center_idx is not None:
                # //  obj_verts3d = obj_verts3d - center3d
                sample[TransQueries.OBJ_VERTS_3D] = obj_verts3d.astype(np.float32)

        # ? ================ Get 3D obj corners
        if BaseQueries.OBJ_CORNERS_3D in query or TransQueries.OBJ_CORNERS_3D in query:
            obj_corners3d = self.get_obj_corners3d(idx)
            if flip:
                obj_corners3d = obj_corners3d.copy()
                obj_corners3d[:, 0] = -obj_corners3d[:, 0]
            if BaseQueries.OBJ_CORNERS_3D in query:
                sample[BaseQueries.OBJ_CORNERS_3D] = obj_corners3d
            if TransQueries.OBJ_CORNERS_3D in query:
                obj_corners3d = rot_mat.dot(obj_corners3d.transpose(1, 0)).transpose()
                # ! CAUTION! may lead to unexpected bug
                # // if self.center_idx is not None:
                # //    obj_corners3d = obj_corners3d - center3d
                sample[TransQueries.OBJ_CORNERS_3D] = obj_corners3d

        # ? ================ Get obj canonical verts
        if BaseQueries.OBJ_CAN_VERTS in query:
            obj_can_verts, obj_can_trans, obj_can_scale = self.get_obj_verts_can(idx)
            if flip:
                obj_can_verts = obj_can_verts.copy()
                obj_can_verts[:, 0] = -obj_can_verts[:, 0]
            sample[BaseQueries.OBJ_CAN_VERTS] = obj_can_verts
            sample[BaseQueries.OBJ_CAN_SCALE] = obj_can_scale
            sample[BaseQueries.OBJ_CAN_TRANS] = obj_can_trans

        # ? ================ Get obj canonical corners
        if BaseQueries.OBJ_CAN_CORNERS in query:
            obj_can_corners = self.get_obj_corners_can(idx)
            if flip:
                obj_can_corners = obj_can_corners.copy()
                obj_can_corners[:, 0] = -obj_can_corners[:, 0]
            sample[BaseQueries.OBJ_CAN_CORNERS] = obj_can_corners
        # ? <<<<<<<<<<<<<<<< END OF ALL 3D POINTS ANNOTATION <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        #  Get hand & obj faces
        if BaseQueries.HAND_FACES in query:
            hand_faces = self.get_hand_faces(idx)
            sample[BaseQueries.HAND_FACES] = hand_faces
        if BaseQueries.OBJ_FACES in query:
            obj_faces = self.get_obj_faces(idx)
            sample[BaseQueries.OBJ_FACES] = obj_faces

        # Get rgb image
        if TransQueries.IMAGE in query:
            # ============== Data augmentation >>>>>>>>>>>>>>
            if self.data_split in ["train", "trainval"]:
                blur_radius = Uniform(low=0, high=1).sample().item() * self.blur_radius
                img = img.filter(ImageFilter.GaussianBlur(blur_radius))
                bright, contrast, sat, hue = augutils.get_color_params(
                    brightness=self.brightness, saturation=self.saturation, hue=self.hue, contrast=self.contrast,
                )
                img = augutils.apply_jitter(img, brightness=bright, saturation=sat, hue=hue, contrast=contrast)

            #  Transform and crop
            img = augutils.transform_img(img, affinetrans, self.inp_res)
            img = img.crop((0, 0, self.inp_res[0], self.inp_res[1]))

            #  Tensorize and normalize_img
            img = func.to_tensor(img).float()
            img = func.normalize(img, [0.5, 0.5, 0.5], [1, 1, 1])
            if TransQueries.IMAGE in query:
                sample[TransQueries.IMAGE] = img

        if BaseQueries.OBJ_TRANSF in query:
            sample[BaseQueries.OBJ_TRANSF] = self.get_obj_transf(idx)
        if TransQueries.OBJ_TRANSF in query:
            base_trasnf = self.get_obj_transf(idx)
            base_rot = base_trasnf[:3, :3]  # (3, 3)
            base_tsl = base_trasnf[:3, 3:]  # (3, 1)
            trans_rot = rot_mat @ base_rot  # (3, 3)
            trans_tsl = rot_mat.dot(base_tsl)  # (3, 1)
            trans_transf = np.concatenate([trans_rot, trans_tsl], axis=1)  # (3, 4)
            trans_transf = np.concatenate([trans_transf, np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)
            sample[TransQueries.OBJ_TRANSF] = trans_transf.astype(np.float32)

        # ! =============== GET CONTACT INFO  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # ! this part currently have nothing to do with augmentation
        # ! may change

        if ContactQueries.HAND_PALM_VERT_IDX in query:
            sample[ContactQueries.HAND_PALM_VERT_IDX] = self.get_hand_palm_vert_idx(idx)

        if (
            ContactQueries.VERTEX_CONTACT in query
            or ContactQueries.CONTACT_REGION_ID in query
            or ContactQueries.CONTACT_ANCHOR_ID in query
            or ContactQueries.CONTACT_ANCHOR_DIST in query
            or ContactQueries.CONTACT_ANCHOR_ELASTI in query
        ):
            processed_dict = self.get_processed_contact_info(idx)
            if ContactQueries.VERTEX_CONTACT in query:
                sample[ContactQueries.VERTEX_CONTACT] = processed_dict["vertex_contact"]
            if ContactQueries.CONTACT_REGION_ID in query:
                sample[ContactQueries.CONTACT_REGION_ID] = processed_dict["hand_region"]
            if ContactQueries.CONTACT_ANCHOR_ID in query:
                sample[ContactQueries.CONTACT_ANCHOR_ID] = processed_dict["anchor_id"]
            if ContactQueries.CONTACT_ANCHOR_DIST in query:
                sample[ContactQueries.CONTACT_ANCHOR_DIST] = processed_dict["anchor_dist"]
            if ContactQueries.CONTACT_ANCHOR_ELASTI in query:
                sample[ContactQueries.CONTACT_ANCHOR_ELASTI] = processed_dict["anchor_elasti"]
            contact_anchor_padding_mask = processed_dict["anchor_padding_mask"]
            if contact_anchor_padding_mask is not None:
                sample[ContactQueries.CONTACT_ANCHOR_PADDING_MASK] = contact_anchor_padding_mask
        # ! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # for offline training only
        sample[BaseQueries.HAND_POSE_WRT_CAM] = self.get_hand_pose_wrt_cam(idx)

        # get sample identifier
        sample[MetaQueries.SAMPLE_IDENTIFIER] = self.get_sample_identifier(idx)

        return sample



@DATASETS.register_module()
class HODataset(HOdata):
    CLASSES=None
    def __init__(self,
                 datasets='ho3d',
                 data_root="data",
                 data_split='train',
                 split_mode='objects',
                 use_cache=False,
                 mini_factor=1,
                 center_idx=9,
                 enable_contact=False,
                 filter_no_contact=False,
                 filter_thresh=0.0,  # useless
                 # query=query,
                 like_v1=True,
                 block_rot=True,
                 max_rot=True,
                 scale_jittering=0,
                 center_jittering=0.1,
                 synt_factor=(0),
                 assets_root='./assets',
                 model_root='./YCB_models',
                 num_points=None,
                 init_cfgs=dict(
                     data_root="data",
                     data_split="train",
                     njoints=21,
                     use_cache=True,
                     filter_no_contact=False,
                     filter_thresh=10.0,
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
                     assets_root='assets'
                 ),
                 pipeline=None,
                 test_mode=False,
                 ):
        super(HODataset, self).__init__(**init_cfgs)
        if not isinstance(datasets, list):
            datasets = [datasets]
        for d in datasets:
            assert d in ["ho3d", "ho3dsynt", "ho3dsynttsl", "ho3dgen", "ho3dycba", "ho3dtrick", "fhb", "fhbsynt"]
        assert split_mode in ["objects", "subjects", "actions", "official"]

        if enable_contact:
            self.all_queries.update(
                {
                    ContactQueries.VERTEX_CONTACT,
                    ContactQueries.CONTACT_REGION_ID,
                    ContactQueries.CONTACT_ANCHOR_ID,
                    ContactQueries.CONTACT_ANCHOR_DIST,
                    ContactQueries.CONTACT_ANCHOR_ELASTI,
                    ContactQueries.CONTACT_ANCHOR_PADDING_MASK,
                }
            )


        self.datasets = []
        for d in datasets:
            self.datasets.append(
                HOdata(**init_cfgs).get_dataset(
                dataset=d,
                data_root=data_root,
                data_split=data_split,
                split_mode=split_mode,
                use_cache=use_cache,
                mini_factor=mini_factor,
                center_idx=center_idx,
                enable_contact=enable_contact,
                filter_no_contact=filter_no_contact,
                filter_thresh=filter_thresh,
                like_v1=like_v1,
                block_rot=block_rot,
                max_rot=max_rot,
                scale_jittering=scale_jittering,
                center_jittering=center_jittering,
                synt_factor=synt_factor,
                query=self.all_queries,
                assets_root=assets_root,
                model_root=model_root
            ))
        assert len(self.datasets) == len(synt_factor), f"train dataset and synt+factor not match"
        self.datasets = ConcatDataset(self.datasets)

        self.test_mode = test_mode
        if self.test_mode == True:
            self.evaluator = Evaluator()
        self.pipeline = pipeline
        if self.pipeline is not None:
            self.pipeline = Compose(self.pipeline)
        self.flag = np.zeros(len(self), dtype=np.int64)

        self.num_points = num_points
        self.padding_keys = [BaseQueries.OBJ_CAN_VERTS, BaseQueries.OBJ_VERTS_3D, BaseQueries.OBJ_VERTS_2D, BaseQueries.OBJ_VIS_2D,
                             TransQueries.OBJ_VERTS_2D, TransQueries.OBJ_VERTS_3D,
                             ContactQueries.CONTACT_ANCHOR_DIST, ContactQueries.CONTACT_ANCHOR_ELASTI, ContactQueries.CONTACT_ANCHOR_ID,
                             ContactQueries.CONTACT_ANCHOR_PADDING_MASK, ContactQueries.CONTACT_REGION_ID, ContactQueries.VERTEX_CONTACT]

    def __getitem__(self, item):
        results = self.datasets.__getitem__(item)
        results = self.processing(results)
        if self.pipeline is not None:
            results = self.pipeline(results)
        return results

    def __len__(self):
        return len(self.datasets)
        # return 10


    def processing(self, results):
        if self.num_points is not None:
            results_processed = {str(k): pc_padding(v, num_points=self.num_points) if k in self.padding_keys else v for k, v in results.items()}
        else:
            results_processed = {str(k): v for k, v in results.items()}
        results_processed['CollateQueries.PADDING_MASK'] = np.ones(self.num_points)
        return results_processed


    def evaluate(self,
                 results,
                 **kwargs
                 ):
        dataloader = DataLoader(self, collate_fn=evaluate_collate)
        for i, data in enumerate(dataloader):
            # data.pop('img_metas')
            data = recover_queries(data)
            self.evaluator.feed_loss_meters(data, results[i])
            self.evaluator.feed_eval_meters(data, results[i])
            if i == len(results):
                break
        eval_results = self.evaluator.parse_evaluators()
        return eval_results




def pc_padding(pc, num_points):
    divisor = pc.shape[0] // num_points
    if divisor > 1:
        pc_pad = pc.repeat(divisor + 1, 0)[:num_points]
    else:
        idx = np.random.choice(pc.shape[0], size=num_points)
        pc_pad = pc[idx]
    return  pc_pad
