from hocontact.visualize.vis_contact_info import create_vertex_color
from math import pi
from manojax.manolayer import ManoLayer
from manojax.anchorlayer import AnchorLayer
from manojax.axislayer import AxisLayer
import numpy

from manojax.quatutils import (
    quaternion_norm_squared,
    normalize_quaternion,
    quaternion_to_angle_axis,
    quaternion_mul,
    quaternion_inv,
)
from jax import nn
import jax.numpy as np
from jax.experimental import optimizers
from jax import value_and_grad
from jax import grad, jit, vmap
import trimesh
from manopth.anchorutils import masking_load_driver, get_region_palm_mask
from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments

from hocontact.hodatasets.hodata import HOdata
from manopth.anchorutils import anchor_load_driver
from tqdm import trange
from termcolor import colored, cprint
import open3d as o3d

from manojax.rodrigues_layer import batch_rodrigues_jax

# from dualhonet.mkgrab.vis_contact_info import create_vertex_color
from manojax.quatutils import angle_axis_to_quaternion


class HandOptimizerJax:
    # * <<<< global layers <<<<<
    mano_layer = ManoLayer(
        joint_rot_mode="quat",
        root_rot_mode="quat",
        use_pca=False,
        mano_root="assets/mano",
        center_idx=None,
        flat_hand_mean=True,
        return_transf=True,
    )
    axis_layer = AxisLayer()
    anchor_layer = AnchorLayer("./data/info/anchor")

    # * initialize optimizer & jax numpy related stuff
    def __init__(
        self, lr=1e-2, n_iter=2500,
    ):
        self.lr = lr
        self.n_iter = n_iter
        self.optimizer = optimizers.adam(step_size=self.lr)

        # * opt val dict, const_val dict
        self.opt_val = {}
        self.const_val = {}

        hand_faces = HandOptimizerJax.mano_layer.faces
        static_verts = self.get_static_hand_verts()
        self.const_val["hand_edges"] = self.get_edge_idx(hand_faces)
        self.const_val["static_edge_len"] = self.get_edge_len(static_verts, self.const_val["hand_edges"])

        # * "just in time" compile loss function here
        self.jit_loss_fn = jit(HandOptimizerJax.loss_fn)

    def set_opt_val(
        self,
        obj_verts_3d,  # JAXNP[NVERT, 3]
        obj_normals,  # JAXNP[NVERT, 3]
        vertex_contact,  # JAXNP[NVERT, ] {0,1}
        contact_region,  # JAXNP[NVERT, 1], int
        anchor_id,  # JAXNP[NVERT, 4]: int
        anchor_elasti,  # JAXNP[NVERT, 4]
        anchor_padding_mask,  # JAXNP[NVERT, 4] {0,1}
        hand_region_assignment,  # JAXNP[NHANDVERT, ]
        hand_palm_vertex_mask,  # JAXNP[NHANDVERT, ] {0, 1}
        trans_gt=None,  # JAXNP[3, ]
        shape_gt=None,  # JAXNP[10, ]
        pose_gt=None,  # (LIST[NPROV, ]: int {0..16}, JAXNP[NPROV, 4])
        init_pose=None,
        opt_obj_transf=False,
        verbose=False,
    ):
        vertex_contact = vertex_contact.astype(np.int32)
        anchor_id = anchor_id.astype(np.int32)
        anchor_padding_mask = anchor_padding_mask.astype(np.int32)

        self.const_val["full_obj_verts_3d"] = obj_verts_3d

        # STEP 1: boolean index objvert3d, anchor_id, anchor_elasti && anchor_padding_mask
        obj_verts_3d = obj_verts_3d[vertex_contact == 1, :]  # JAXNP[NCONT, 3]
        obj_normals = obj_normals[vertex_contact == 1, :]  # JAXNP[NCONT, 3]
        obj_contact_region = contact_region[vertex_contact == 1]  # JAXNP[NCONT, ]
        anchor_id = anchor_id[vertex_contact == 1, :]  # JAXNP[NCONT, 4]
        anchor_elasti = anchor_elasti[vertex_contact == 1, :]  # JAXNP[NCONT, 4]
        anchor_padding_mask = anchor_padding_mask[vertex_contact == 1, :]  # JAXNP[NCONT, 4]

        # STEP 2: boolean mask indexing anchor_id, anchor_elasti && object_vert_id
        self.const_val["indexed_anchor_id"] = anchor_id[anchor_padding_mask == 1]  # JAXNP[NVALID, ]
        self.const_val["indexed_anchor_elasti"] = anchor_elasti[anchor_padding_mask == 1]  # JAXNP[NVALID, ]

        vertex_id = np.broadcast_to(
            np.arange(anchor_id.shape[0])[:, None], (anchor_id.shape[0], anchor_padding_mask.shape[1]),
        )  # JAXNP[NCONT, 4]
        indexed_vertex_id = vertex_id[anchor_padding_mask == 1]  # JAXNP[NVALID, ]
        self.const_val["indexed_elasti_k"] = np.where(
            np.isin(
                self.const_val["indexed_anchor_id"], np.array([2, 3, 4, 9, 10, 11, 15, 16, 17, 22, 23, 24, 29, 30, 31]),
            ),
            5.0,
            1.0,
        )

        # STEP 3: index anchor && object vert
        self.const_val["vertex_pos"] = obj_verts_3d[indexed_vertex_id]  # JAXNP[NVALID, 3]

        # STEP 4: prepare essentials for repulsion loss
        obj_vert_idx_list = []
        hand_vert_idx_list = []
        for vertex_id, contact_region_id in enumerate(obj_contact_region):
            selected_hand_vert_mask = get_region_palm_mask(
                contact_region_id, None, hand_region_assignment, hand_palm_vertex_mask
            )
            selected_hand_vert_idx = np.where(selected_hand_vert_mask)[0]
            repeat_times = selected_hand_vert_idx.shape[0]
            obj_vertex_idx = np.ones((repeat_times,), dtype=np.int32) * vertex_id
            obj_vert_idx_list.append(obj_vertex_idx)
            hand_vert_idx_list.append(selected_hand_vert_idx)
        self.const_val["concat_hand_vert_idx"] = np.concatenate(hand_vert_idx_list, axis=0)
        concat_obj_vert_idx = np.concatenate(obj_vert_idx_list, axis=0)
        self.const_val["concat_obj_vert_3d"] = obj_verts_3d[concat_obj_vert_idx, :]
        self.const_val["concat_obj_normal"] = obj_normals[concat_obj_vert_idx, :]

        # STEP 5: initialize all variables
        if trans_gt is None:  # optimize tsl
            self.opt_val["hand_trans"] = np.zeros(3, dtype=np.float32)
        else:
            self.const_val["hand_trans"] = np.array(trans_gt, copy=True)
        if shape_gt is None:  # optimize shape
            self.opt_val["hand_shape"] = np.zeros(10, dtype=np.float32)
        else:
            self.const_val["hand_shape"] = np.array(shape_gt, copy=True)

        if pose_gt is None:
            self.const_val["sel_pose_idx"], self.const_val["sel_pose_gt"] = (
                [],
                np.zeros((0, 4), dtype=np.float32),
            )
        else:
            self.const_val["sel_pose_idx"], self.const_val["sel_pose_gt"] = pose_gt

        self.const_val["var_pose_idx"] = HandOptimizerJax.get_var_pose_idx(self.const_val["sel_pose_idx"])
        n_var_pose = len(self.const_val["var_pose_idx"])

        if n_var_pose > 0:
            init_val = np.array([[0.9999, 0.0, -0.0101, 0.0]] * n_var_pose).astype(np.float32)
            self.opt_val["hand_pose"] = init_pose if init_pose is not None else init_val
        else:
            self.const_val["hand_pose"] = np.zeros((0, 4))

        if opt_obj_transf:
            self.opt_val["obj_tsl"] = np.array([0.001, 0.001, 0.001]).astype(np.float32)
            self.opt_val["obj_rot"] = np.array([0.001, 0.001, 0.001]).astype(np.float32)
        else:
            self.const_val["obj_tsl"] = np.zeros(3).astype(np.float32)
            self.const_val["obj_rot"] = np.zeros(3).astype(np.float32)

        optim_flag = len(self.const_val) > 0 or opt_obj_transf
        if verbose:
            print("Optimize: ", optim_flag)
            print("Optimize: ", self.const_val["var_pose_idx"])

    @staticmethod
    def get_var_pose_idx(sel_pose_idx):
        # gt has 16 pose
        all_pose_idx = set(range(16))
        sel_pose_idx_set = set(sel_pose_idx)
        var_pose_idx = all_pose_idx.difference(sel_pose_idx_set)
        return list(var_pose_idx)

    @staticmethod
    def get_static_hand_verts():
        init_val_pose = np.array([[1.0, 0.0, 0.0, 0.0]] * 16).astype(np.float32)
        vec_pose = np.array(init_val_pose).reshape(-1)[None]
        vec_shape = np.zeros((1, 10), dtype=np.float32)
        v, _, _ = HandOptimizerJax.mano_layer(vec_pose, vec_shape)
        v = v.squeeze(0)
        return v

    # todo: validate correctness of this function
    @staticmethod
    def get_edge_idx(face_idx_jnp):
        res = []
        face_idx_list = face_idx_jnp.astype(np.float32).tolist()
        for item in face_idx_list:
            v_idx_0, v_idx_1, v_idx_2 = item
            if {v_idx_0, v_idx_1} not in res:
                res.append({v_idx_0, v_idx_1})
            if {v_idx_1, v_idx_2} not in res:
                res.append({v_idx_1, v_idx_2})
            if {v_idx_0, v_idx_2} not in res:
                res.append({v_idx_0, v_idx_2})
        res = [list(e) for e in res]
        return np.array(res, dtype=np.int32)

    @staticmethod
    def get_edge_len(verts, edge_idx):
        # verts: JNP[NVERT, 3]
        # edge_idx: JNP[NEDGE, 2]
        return np.linalg.norm(verts[edge_idx[:, 0], :] - verts[edge_idx[:, 1], :], axis=1)

    @staticmethod
    def assemble_pose_vec(gt_idx, gt_pose, var_idx, var_pose):
        idx_array = np.concatenate((np.array(gt_idx), np.array(var_idx)))
        pose_array = np.concatenate((gt_pose.reshape(len(gt_idx), 4), var_pose.reshape(len(var_idx), 4)))
        pose_array = pose_array[np.argsort(idx_array)]
        return pose_array.reshape(-1)

    # *************************** losses ***************************
    @staticmethod
    def pose_quat_norm_loss(var_pose):
        """ this is the only loss accepts unnormalized quats """
        reshaped_var_pose = var_pose.reshape((16, 4))  # JNP[16, 4]
        quat_norm_sq = quaternion_norm_squared(reshaped_var_pose)  # JNP[16, ]
        squared_norm_diff = quat_norm_sq - 1.0  # JNP[16, ]
        res = np.mean(np.power(squared_norm_diff, 2), axis=0)
        return res

    @staticmethod
    def pose_reg_loss(var_pose_normed):
        # the format of quat is [w, x, y, z]
        # to regularize
        # just to make sure w is close to 1.0
        # working aside with self.pose_quat_norm_loss defined above
        reshaped_var_pose_normed = var_pose_normed.reshape((16, 4))
        w = reshaped_var_pose_normed[..., 0]  # get w
        diff = w - 1.0  # JNP[16, ]
        res = np.mean(np.power(diff, 2), axis=0)
        return res

    @staticmethod
    def joint_smooth_loss(var_pose_normed):
        lvl1_idxs = [1, 4, 7, 10, 13]
        lvl2_idxs = [2, 5, 8, 11, 14]
        lvl3_idxs = [3, 6, 9, 12, 15]
        reshaped_var_pose_normed = var_pose_normed.reshape((16, 4))  # JNP[16, 4]
        lvl1_quat = reshaped_var_pose_normed[lvl1_idxs, :]  # JNP[5, 4]
        lvl2_quat = reshaped_var_pose_normed[lvl2_idxs, :]  # JNP[5, 4]
        lvl3_quat = reshaped_var_pose_normed[lvl3_idxs, :]  # JNP[5, 4]
        # first part, lvl2 -> lvl1
        combinated_quat_real_1 = quaternion_mul(quaternion_inv(lvl2_quat), lvl1_quat)[..., 0]  # JNP[5, ]
        combinated_quat_real_diff_1 = combinated_quat_real_1 - 1.0
        loss_part_1 = np.mean(np.power(combinated_quat_real_diff_1, 2), axis=0)
        # second part, lvl3 -> lvl2
        combinated_quat_real_2 = quaternion_mul(quaternion_inv(lvl3_quat), lvl2_quat)[..., 0]  # JNP[5, ]
        combinated_quat_real_diff_2 = combinated_quat_real_2 - 1.0
        loss_part_2 = np.mean(np.power(combinated_quat_real_diff_2, 2), axis=0)
        # combine them
        res = loss_part_1 + loss_part_2
        return res

    @staticmethod
    def rotation_angle_loss(angle, limit_angle=pi / 2, eps=1e-10):
        # nonzero_mask = np.abs(angle) > eps  # JNP[15, ], bool
        # angle_new = angle[nonzero_mask]  # if angle is too small, pick them out of backward graph
        angle_over_limit = nn.relu(angle - limit_angle)  # < pi/2, 0; > pi/2, linear | JNP[16, ]
        angle_over_limit_squared = np.power(angle_over_limit, 2)  # JNP[15, ]
        res = np.mean(angle_over_limit_squared, axis=0)
        return res

    # ???? axis order right hand

    #         14-13-12-\
    #                   \
    #    2-- 1 -- 0 -----*
    #   5 -- 4 -- 3 ----/
    #   11 - 10 - 9 ---/
    #    8-- 7 -- 6 --/

    @staticmethod
    def joint_b_axis_loss(b_axis, axis):
        b_soft_idx = [0, 3, 9, 6]
        b_thumb_soft_idx = 12
        b_axis = b_axis.squeeze(0)  # [15, 3]

        b_axis_cos = (np.expand_dims(b_axis, axis=1) @ np.expand_dims(axis, axis=2)).squeeze(1).squeeze(1)
        restrict_cos = b_axis_cos[[i for i in range(15) if i not in b_soft_idx and i != b_thumb_soft_idx]]
        soft_loss = nn.relu(np.abs(b_axis_cos[b_soft_idx]) - np.cos(pi / 2 - pi / 36))  # [-5, 5]
        thumb_soft_loss = nn.relu(
            np.abs(np.expand_dims(b_axis_cos[b_thumb_soft_idx], axis=0)) - np.cos(pi / 2 - pi / 9)  # [-20, 20]
        )
        res = (
            np.mean(np.power(restrict_cos, 2), axis=0)
            + np.mean(np.power(soft_loss, 2), axis=0)
            + np.mean(np.power(thumb_soft_loss, 2), axis=0)
        )
        return res

    @staticmethod
    def joint_u_axis_loss(u_axis, axis):
        u_soft_idx = [0, 3, 9, 6]
        u_thumb_soft_idx = 12
        u_axis = u_axis.squeeze(0)  # [15, 3]

        u_axis_cos = (np.expand_dims(u_axis, axis=1) @ np.expand_dims(axis, axis=2)).squeeze(1).squeeze(1)
        restrict_cos = u_axis_cos[[i for i in range(15) if i not in u_soft_idx and i != u_thumb_soft_idx]]
        soft_loss = nn.relu(np.abs(u_axis_cos[u_soft_idx]) - np.cos(pi / 2 - pi / 18))  # [-10, 10]
        thumb_soft_loss = nn.relu(
            np.abs(np.expand_dims(u_axis_cos[u_thumb_soft_idx], axis=0)) - np.cos(pi / 2 - pi / 9)  # [-20, 20]
        )
        res = (
            np.mean(np.power(restrict_cos, 2), axis=0)
            + np.mean(np.power(soft_loss, 2), axis=0)
            + np.mean(np.power(thumb_soft_loss, 2), axis=0)
        )
        return res

    @staticmethod
    def joint_l_limit_loss(l_axis, axis):
        l_soft_idx = [0, 3, 9, 6]
        l_thumb_soft_idx = 12
        l_axis = l_axis.squeeze(0)  # [15, 3]
        l_axis_cos = (np.expand_dims(l_axis, axis=1) @ np.expand_dims(axis, axis=2)).squeeze(1).squeeze(1)
        restrict_cos = l_axis_cos[[i for i in range(15) if i not in l_soft_idx and i != l_thumb_soft_idx]]
        soft_loss = nn.relu(-l_axis_cos[l_soft_idx] + 1 - np.cos(pi / 2 - pi / 18))
        thumb_soft_loss = nn.relu(-np.expand_dims(l_axis_cos[l_thumb_soft_idx], axis=0) + 1 - np.cos(pi / 2 - pi / 9))

        res = (
            np.mean(np.power(restrict_cos - 1, 2), axis=0)
            + np.mean(np.power(soft_loss, 2), axis=0)
            + np.mean(np.power(thumb_soft_loss, 2), axis=0)
        )
        return res

    @staticmethod
    def contact_loss(apos, vpos, e, e_k):
        # apos, vpos = JNP[NVALID, 3]
        # e = JNP[NVALID, ]
        sqrt_e = np.sqrt(e) * e_k
        dist = np.sum(np.power(vpos - apos, 2), axis=1)  # JNP[NVALID, ]
        res = np.mean(sqrt_e * dist, axis=0)
        return res

    @staticmethod
    def edge_len_loss(rebuild_verts, hand_edges, static_edge_len):
        pred_edge_len = HandOptimizerJax.get_edge_len(rebuild_verts, hand_edges)
        diff = pred_edge_len - static_edge_len  # JNP[NEDGE, ]
        return np.mean(np.power(diff, 2), axis=0)

    @staticmethod
    def hand_back_repulsion_direction_loss(
        pred_hand_verts, concat_hand_vert_idx, concat_obj_vert_3d, concat_obj_normal,
    ):
        # pred_hand_verts = JNP[NHANDVERTS, 3]
        selected_hand_verts = pred_hand_verts[concat_hand_vert_idx, :]  # JNP[NCC, 3]
        # compute offset vector from object to hand
        offset_vectors = selected_hand_verts - concat_obj_vert_3d  # JNP[NCC, 3]
        # compute inner product (not normalized)
        inner_product = np.expand_dims(offset_vectors, 1) @ np.expand_dims(concat_obj_normal, 2)
        # JNP[NCC, 1, 3] @ JNP[NCC, 3, 1] -> JNP[NCC, 1, 1]
        inner_product = inner_product.squeeze(2).squeeze(1)  # JNP[NCC, ]

        thresholded_value = np.exp(nn.relu(-inner_product)) - 1  # JNP[NCC, ]
        # res = torch.mean(torch.pow(thresholded_value, 2), dim=0)
        res = np.mean(thresholded_value, axis=0)
        return res

    @staticmethod
    def repulsion_loss(
        pred_hand_verts, concat_hand_vert_idx, concat_obj_vert_3d, concat_obj_normal, constant=0.05, threshold=0.01,
    ):
        # pred_hand_verts = JNP[NHANDVERTS, 3]
        selected_hand_verts = pred_hand_verts[concat_hand_vert_idx, :]  # JNP[NCC, 3]
        # compute offset vector from object to hand
        offset_vectors = selected_hand_verts - concat_obj_vert_3d  # JNP[NCC, 3]
        # compute inner product (not normalized)
        inner_product = np.expand_dims(offset_vectors, 1) @ np.expand_dims(concat_obj_normal, 2)
        # JNP[NCC, 1, 3] @ JNP[NCC, 3, 1] -> JNP[NCC, 1, 1]
        inner_product = inner_product.squeeze(2).squeeze(1)  # JNP[NCC, ]

        thresholded_value = constant * np.power(np.exp(nn.relu(-inner_product + threshold)) - 1, 2)
        # res = torch.mean(torch.pow(thresholded_value, 2), dim=0)
        res = np.sum(thresholded_value, axis=0)
        return res

    @staticmethod
    def transf_vectors(vectors, tsl, rot):
        """
        vectors: [K, 3], tsl: [3, ], rot: [3, ]
        return: [K, 3]
        """
        rot_matrix = batch_rodrigues_jax(np.expand_dims(rot, axis=0)).squeeze(0)
        vec = (rot_matrix @ vectors.T).T
        vec = vec + tsl
        return vec

    @staticmethod
    def obj_transf_loss(obj_tsl, obj_rot):
        tsl_loss = np.power((obj_tsl - np.zeros(3, dtype=np.float32)), 2)
        rot_loss = np.power((obj_rot - np.zeros(3, dtype=np.float32)), 2)
        return np.mean(tsl_loss, axis=0) + np.mean(rot_loss, axis=0)

    # ! >>>>>>>> deprecated >>>>>>>>
    @staticmethod
    def collision_loss_hand_in_obj(
        hand_verts, obj_verts, obj_faces,
    ):
        # unsqueeze fist dimension so that we can use hasson's utils directly
        # todo: reimplement this in non batch way
        hand_verts = np.expand_dims(hand_verts, 0)
        obj_verts = np.expand_dims(obj_verts, 0)

        # Get obj triangle positions
        obj_triangles = obj_verts[:, obj_faces]
        exterior = batch_mesh_contains_points(hand_verts, obj_triangles)
        penetr_mask = ~exterior

        # only compute exterior related stuff
        valid_vals = penetr_mask.sum()
        # if valid_vals > 0:
        selected_hand_verts = hand_verts[penetr_mask, :]
        selected_hand_verts = np.expand_dims(selected_hand_verts, 0)
        dists = batch_pairwise_dist(selected_hand_verts, obj_verts)
        mins_sel_hand_to_obj = np.min(dists, 2)
        mins_sel_hand_to_obj_idx = np.argmin(dists, 2)
        results_close = batch_index_select(obj_verts, mins_sel_hand_to_obj_idx)
        collision_vals = ((results_close - selected_hand_verts) ** 2).sum(2)

        penetr_loss = np.mean(collision_vals)
        # else:
        #     penetr_loss = np.mean(np.array([0.0]), axis=0)
        return penetr_loss

    # ! <<<<<<<< deprecated <<<<<<<<

    # *>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    @staticmethod
    def loss_fn(opt_val, const_val):

        cprint("compile!!", "green")

        # * rename optimizer
        HOJ = HandOptimizerJax

        vars_hand_pose = opt_val["hand_pose"] if "hand_pose" in opt_val else const_val["hand_pose"]
        vars_hand_trans = opt_val["hand_trans"] if "hand_trans" in opt_val else const_val["hand_trans"]
        vars_hand_shape = opt_val["hand_shape"] if "hand_shape" in opt_val else const_val["hand_shape"]
        vars_obj_tsl = opt_val["obj_tsl"] if "obj_tsl" in opt_val else const_val["obj_tsl"]
        vars_obj_rot = opt_val["obj_rot"] if "obj_rot" in opt_val else const_val["obj_rot"]

        #  ========== layers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        vars_hand_pose_assembled = HOJ.assemble_pose_vec(
            const_val["sel_pose_idx"], const_val["sel_pose_gt"], const_val["var_pose_idx"], vars_hand_pose,
        )  # JNP[64, ]
        # quat basic normalization loss (soft constraint for unit quat)
        quat_norm_loss = HOJ.pose_quat_norm_loss(vars_hand_pose_assembled)  # JNP[]
        # normalize the quat
        vars_hand_pose_normalized = normalize_quaternion(vars_hand_pose_assembled.reshape((16, 4))).reshape(-1)
        # JNP[64, ]
        # quat zero rotation loss
        pose_reg_loss = HOJ.pose_reg_loss(vars_hand_pose_normalized)
        #
        # unsqueeze 0 dimension to form a batch with size 1
        vec_pose = np.expand_dims(vars_hand_pose_assembled, axis=0)
        vec_shape = np.expand_dims(vars_hand_shape, axis=0)
        vec_trans = np.expand_dims(vars_hand_trans, axis=0)
        rebuild_verts, rebuild_joints, rebuild_transf = HOJ.mano_layer(vec_pose, vec_shape)
        rebuild_joints = rebuild_joints + vec_trans
        rebuild_verts = rebuild_verts + vec_trans
        rebuild_transf = rebuild_transf + np.concatenate(
            (np.concatenate((np.zeros((3, 3), dtype=np.float32), vec_trans.reshape(3, -1)), 1), np.zeros((1, 4)),), 0,
        )

        rebuild_verts_squeezed = rebuild_verts.squeeze(0)

        b_axis, u_axis, l_axis = HOJ.axis_layer(rebuild_joints, rebuild_transf)
        # ***** calculate losses on back, up, left axis
        angle_axis_with_root = quaternion_to_angle_axis(vars_hand_pose_normalized.reshape((16, 4)))
        angle_axis = angle_axis_with_root[1:, :]  # ignore global rot [15, 3]
        angle = np.expand_dims(np.linalg.norm(angle_axis, axis=1), -1)
        axis = angle_axis / angle
        # limit angle
        angle_limit_loss = HOJ.rotation_angle_loss(angle.squeeze(-1))

        joint_b_axis_loss = HOJ.joint_b_axis_loss(b_axis, axis)
        joint_u_axis_loss = HOJ.joint_u_axis_loss(u_axis, axis)
        joint_l_limit_loss = HOJ.joint_l_limit_loss(l_axis, axis)

        rebuild_anchor = HOJ.anchor_layer(rebuild_verts)
        # rebuild_anchor = rebuild_anchor.contiguous()  # JNP[1, 32, 3]
        rebuild_anchor = rebuild_anchor.squeeze(0)  # JNP[32, 3]

        # index rebuild_anchor
        anchor_pos = rebuild_anchor[const_val["indexed_anchor_id"]]  # JNP[NVALID, 3]
        contact_loss = HOJ.contact_loss(
            anchor_pos,
            HOJ.transf_vectors(const_val["vertex_pos"], vars_obj_tsl, vars_obj_rot),
            const_val["indexed_anchor_elasti"],
            const_val["indexed_elasti_k"],
        )

        edge_loss = HOJ.edge_len_loss(rebuild_verts_squeezed, const_val["hand_edges"], const_val["static_edge_len"],)
        repulsion_loss = HOJ.repulsion_loss(
            rebuild_verts_squeezed,
            const_val["concat_hand_vert_idx"],
            HOJ.transf_vectors(const_val["concat_obj_vert_3d"], vars_obj_tsl, vars_obj_rot),
            HOJ.transf_vectors(const_val["concat_obj_normal"], vars_obj_tsl, vars_obj_rot),
        )

        obj_transf_loss = HOJ.obj_transf_loss(vars_obj_tsl, vars_obj_rot)
        # get loss
        loss = (
            # HAND SELF LOSS
            1.0 * quat_norm_loss
            + 1.0 * pose_reg_loss
            + 100.0 * angle_limit_loss
            + 1.0 * edge_loss
            #
            + 50.0 * joint_b_axis_loss
            + 50.0 * joint_u_axis_loss
            + 50.0 * joint_l_limit_loss
            #
            # # CONTACT LOSS
            + 1000.0 * contact_loss
            + 10.0 * repulsion_loss
            + 10.0 * obj_transf_loss
        )
        return (
            loss,
            {
                "quat_norm_loss": quat_norm_loss,
                "pose_reg_loss": pose_reg_loss,
                "angle_limit_loss": angle_limit_loss,
                "edge_loss": edge_loss,
                "joint_b_axis_loss": joint_b_axis_loss,
                "joint_u_axis_loss": joint_u_axis_loss,
                "joint_l_limit_loss": joint_l_limit_loss,
                "contact_loss": contact_loss,
                "repulsion_loss": repulsion_loss,
                "obj_transf_loss": obj_transf_loss,
            },
        )

    # * <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def optimize(self, progress=False):
        if progress:
            bar = trange(self.n_iter, position=0)
            bar_hand = trange(0, position=1, bar_format="{desc}")
            bar_contact = trange(0, position=2, bar_format="{desc}")
            bar_axis = trange(0, position=3, bar_format="{desc}")
        else:
            bar = range(self.n_iter)

        opt_state = self.optimizer.init_fn(self.opt_val)
        for opt_iter in bar:

            value, grads = value_and_grad(self.jit_loss_fn, has_aux=True)(
                self.optimizer.params_fn(opt_state), self.const_val
            )
            opt_state = self.optimizer.update_fn(opt_iter, grads, opt_state)
            if progress:
                bar.set_description("TOTAL LOSS {:4e}".format(value[0].item()))
                auxiliary = value[1]
                bar_hand.set_description(
                    colored("HAND_REGUL_LOSS: ", "yellow")
                    + "QN={:.3e} PR={:.3e} Edge={:.3e}".format(
                        auxiliary["quat_norm_loss"].item(),  # QN
                        auxiliary["pose_reg_loss"].item(),  # PR
                        # joint_smooth_loss.item(),  # JS
                        # roll_over_loss.item(),  # RO
                        auxiliary["edge_loss"].item(),  # Edge
                        # inv_rot_penalty_loss.item(),  # IR
                        # rad_rot_penalty_loss.item(),  # RR
                        # gt_vertices_loss.item()  # GV
                    )
                )

                bar_contact.set_description(
                    colored("HO_CONTACT_LOSS: ", "blue")
                    + "Conta={:.3e}, Repul={:.3e} OT={:.3e}".format(
                        auxiliary["contact_loss"].item(),
                        auxiliary["repulsion_loss"].item(),  # Conta  # Repul
                        auxiliary["obj_transf_loss"].item(),
                    )
                )
                bar_axis.set_description(
                    colored("ANGLE_LOSS: ", "cyan")
                    + "AL={:.3e} JB{:.3e} JU{:.3e} JL{:.3e}".format(
                        auxiliary["angle_limit_loss"].item(),  # AL
                        auxiliary["joint_b_axis_loss"].item(),  # JB
                        auxiliary["joint_u_axis_loss"].item(),  # JU
                        auxiliary["joint_l_limit_loss"].item(),  # JL
                    )
                )

        return opt_state

    def recover_obj(self, obj_verts, obj_tsl=None, obj_rot=None):
        return self.transf_vectors(
            obj_verts,
            obj_tsl if obj_tsl is not None else self.const_val["obj_tsl"],
            obj_rot if obj_rot is not None else self.const_val["obj_rot"],
        )

    def recover_hand(self, squeeze_out=True, hand_pose=None, hand_shape=None, hand_trans=None):
        vars_hand_pose = hand_pose if hand_pose is not None else self.const_val["hand_pose"]
        vars_hand_trans = hand_trans if hand_trans is not None else self.const_val["hand_trans"]
        vars_hand_shape = hand_shape if hand_shape is not None else self.const_val["hand_shape"]
        vars_hand_pose_assembled = self.assemble_pose_vec(
            self.const_val["sel_pose_idx"],
            self.const_val["sel_pose_gt"],
            self.const_val["var_pose_idx"],
            vars_hand_pose,
        )
        vars_hand_pose_normalized = normalize_quaternion(vars_hand_pose_assembled.reshape((16, 4))).reshape(-1)
        vec_pose = np.expand_dims(vars_hand_pose_normalized, axis=0)
        vec_shape = np.expand_dims(vars_hand_shape, axis=0)
        vec_trans = np.expand_dims(vars_hand_trans, axis=0)
        rebuild_verts, rebuild_joints, rebuild_transf = self.mano_layer(vec_pose, vec_shape)
        rebuild_joints = rebuild_joints + vec_trans
        rebuild_verts = rebuild_verts + vec_trans
        rebuild_transf = rebuild_transf + np.concatenate(
            (np.concatenate((np.zeros((3, 3), dtype=np.float32), vec_trans.reshape(3, -1)), 1), np.zeros((1, 4)),), 0,
        )
        if squeeze_out:
            rebuild_verts, rebuild_joints, rebuild_transf = (
                rebuild_verts.squeeze(0),
                rebuild_joints.squeeze(0),
                rebuild_transf.squeeze(0),
            )
        return rebuild_verts, rebuild_joints, rebuild_transf


def main(args):
    hodata = HOdata.get_dataset(
        dataset="ho3d",
        data_root="data",
        data_split="test",
        split_mode="objects",
        use_cache=True,
        mini_factor=1.0,
        center_idx=9,
        enable_contact=True,
        like_v1=True,
        filter_no_contact=False,
        filter_thresh=10,
        block_rot=True,
        load_objects_reduced=True,
    )

    if args.vis:
        hand_mesh_pred = o3d.geometry.TriangleMesh()
        obj_mesh_pred = o3d.geometry.TriangleMesh()
        obj_mesh_gt = o3d.geometry.TriangleMesh()
        hand_mesh_gt = o3d.geometry.TriangleMesh()
        vis_pred = o3d.visualization.Visualizer()
        vis_pred.create_window(window_name="Predicted Hand", width=640, height=640)
        vis_gt = o3d.visualization.Visualizer()
        vis_gt.create_window(window_name="Ground-Truth Hand", width=640, height=640)
        vis_pred.add_geometry(hand_mesh_pred)
        vis_pred.add_geometry(obj_mesh_pred)
        vis_gt.add_geometry(hand_mesh_gt)
        vis_gt.add_geometry(obj_mesh_gt)

    hoptim = HandOptimizerJax(lr=args.lr, n_iter=args.iters)

    for _ in range(100):
        idx = numpy.random.randint(0, len(hodata))
        cprint(idx, "red")
        hand_gt = hodata.get_hand_verts3d(idx)
        contact_info = hodata.get_processed_contact_info(idx)

        obj_mesh = trimesh.Trimesh(hodata.get_obj_verts_transf(idx), hodata.get_obj_faces(idx), process=False)
        obj_vertex_normals_np = obj_mesh.vertex_normals

        s_obj_verts_3d = np.array(hodata.get_obj_verts_transf(idx), dtype=np.float32)
        s_obj_normals = np.array(obj_vertex_normals_np, dtype=np.float32)
        s_obj_faces = np.array(obj_mesh.faces, dtype=np.int32)
        s_hand_verts_3d = np.array(hand_gt, dtype=np.float32)
        s_vertex_contact = np.array(contact_info["vertex_contact"], dtype=np.int32)
        s_contact_region = np.array(contact_info["hand_region"], dtype=np.int32)
        s_anchor_id = np.array(contact_info["anchor_id"], dtype=np.int32)
        s_anchor_elasti = np.array(contact_info["anchor_elasti"], dtype=np.float32)
        s_anchor_padding_mask = np.array(contact_info["anchor_padding_mask"], dtype=np.int32)

        hra, hpv = masking_load_driver("data/info/anchor", "data/info/hand_palm_full.txt")

        s_hand_region_assignment = np.array(hra, dtype=np.int32)
        s_hand_palm_vertex_mask = np.array(hpv, dtype=np.int32)
        s_tsl_gt = np.array(hodata.get_hand_tsl_wrt_cam(idx), dtype=np.float32)
        s_pose_gt = np.array(hodata.get_hand_pose_wrt_cam(idx), dtype=np.float32)
        s_pose_gt_quat = angle_axis_to_quaternion(s_pose_gt.reshape(-1, 3))

        hoptim.set_opt_val(
            s_obj_verts_3d,
            s_obj_normals,
            s_vertex_contact,
            s_contact_region,
            s_anchor_id,
            s_anchor_elasti,
            s_anchor_padding_mask,
            s_hand_region_assignment,
            s_hand_palm_vertex_mask,
            trans_gt=s_tsl_gt,
            shape_gt=np.zeros(10, dtype=np.float32),
            pose_gt=None,
            init_pose=s_pose_gt_quat,
            opt_obj_transf=True,
            verbose=True,
        )

        opt_state = hoptim.optimize(progress=True)
        params = hoptim.optimizer.params_fn(opt_state)
        vertices_pred, _, _ = hoptim.recover_hand(
            hand_pose=params.get("hand_pose"), hand_shape=params.get("hand_shape"), hand_trans=params.get("hand_trans"),
        )
        obj_pred = hoptim.recover_obj(
            hodata.get_obj_verts_transf(idx), obj_tsl=params.get("obj_tsl"), obj_rot=params.get("obj_rot"),
        )
        print(params.get("obj_tsl"), params.get("obj_rot"))
        obj_faces_full = hodata.get_obj_faces(idx)
        obj_gt = hodata.get_obj_verts_transf(idx)
        mean_vert_err = np.mean(np.linalg.norm(s_hand_verts_3d - vertices_pred, axis=1)).item()
        print("mean vert error", mean_vert_err)

        if args.vis:
            vertices_gt = np.expand_dims(s_hand_verts_3d, 0)
            vertices_pred = numpy.array(vertices_pred)

            # ====================== predicted hand + gt obj >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            hand_faces = hodata.get_hand_faces(idx)

            hand_mesh_pred.triangles = o3d.utility.Vector3iVector(hand_faces)
            hand_mesh_pred.vertices = o3d.utility.Vector3dVector(vertices_pred)
            hand_mesh_pred.compute_vertex_normals()

            obj_mesh_pred.triangles = o3d.utility.Vector3iVector(obj_faces_full)
            obj_mesh_pred.vertices = o3d.utility.Vector3dVector(obj_pred)
            obj_mesh_pred.compute_vertex_normals()
            obj_colors = create_vertex_color(contact_info, mode="contact_region")
            obj_mesh_pred.vertex_colors = o3d.utility.Vector3dVector(obj_colors)

            # ======================== gt hand + gt obj >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            hand_mesh_gt.triangles = o3d.utility.Vector3iVector(hand_faces)
            hand_mesh_gt.vertices = o3d.utility.Vector3dVector(vertices_gt.squeeze(0))
            hand_mesh_gt.compute_vertex_normals()
            hand_mesh_gt.paint_uniform_color([55 / 255.0, 52 / 255.0, 193 / 255.0])

            obj_mesh_gt.triangles = o3d.utility.Vector3iVector(obj_faces_full)
            obj_mesh_gt.vertices = o3d.utility.Vector3dVector(obj_gt)
            obj_mesh_gt.compute_vertex_normals()
            obj_colors = create_vertex_color(contact_info, mode="contact_region")
            obj_mesh_gt.vertex_colors = o3d.utility.Vector3dVector(obj_colors)

            vis_pred.update_geometry(hand_mesh_pred)
            vis_pred.update_geometry(obj_mesh_pred)
            vis_pred.reset_view_point(True)
            vis_gt.update_geometry(hand_mesh_gt)
            vis_gt.update_geometry(obj_mesh_gt)
            vis_gt.reset_view_point(True)

            while True:

                vis_pred.update_geometry(hand_mesh_pred)
                vis_pred.update_geometry(obj_mesh_pred)
                vis_gt.update_geometry(hand_mesh_gt)
                vis_gt.update_geometry(obj_mesh_gt)

                vis_pred.update_renderer()
                vis_gt.update_renderer()

                vis_pred.poll_events()
                vis_gt.poll_events()

                if not vis_pred.poll_events() or not vis_gt.poll_events():
                    break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JAX Hand Manipulation")
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--iters", default=1000, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)

    args = parser.parse_args()
    main(args)
