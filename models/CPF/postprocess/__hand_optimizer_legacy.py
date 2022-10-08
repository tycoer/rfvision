from math import pi
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from manopth.anchorlayer import AnchorLayer
from manopth.anchorutils import masking_load_driver, get_region_palm_mask
from manopth.demo import display_hand
from manopth.manolayer import ManoLayer
from manopth.quatutils import (
    quaternion_norm_squared,
    normalize_quaternion,
    quaternion_inv,
    quaternion_mul,
    quaternion_to_angle_axis,
    angle_axis_to_quaternion,
)
from termcolor import colored
from tqdm import trange
from hocontact.hodatasets.hodata import HOdata
from hocontact.hodatasets.hoquery import BaseQueries, ContactQueries
from hocontact.utils.collisionutils import (
    batch_index_select,
    batch_pairwise_dist,
    batch_mesh_contains_points,
    masked_mean_loss,
)


class HandOptimizer:
    def __init__(
        self,
        device,
        obj_verts_3d,  # TENSOR[NVERT, 3]
        obj_normals,  # TENSOR[NVERT, 3]
        obj_faces,  # TENSOR[NFACES, 3], int
        obj_verts_3d_reduced,  # TENSOR[REDUCE_VER, 3]
        obj_faces_reduced,  # TEMSOR[FACE_REDUCE_VER, 3], int
        vertex_contact,  # TENSOR[NVERT, ] {0,1}
        contact_region,  # TENSOR[NVERT, 1], int
        anchor_id,  # TENSOR[NVERT, 4]: int
        anchor_elasti,  # TENSOR[NVERT, 4]
        anchor_padding_mask,  # TENSOR[NVERT, 4] {0,1}
        hand_region_assignment,  # TENSOR[NHANDVERT, ]
        hand_palm_vertex_mask,  # TENSOR[NHANDVERT, ] {0, 1}
        trans_gt=None,  # TENSOR[3, ]
        shape_gt=None,  # TENSOR[10, ]
        pose_gt=None,  # (LIST[NPROV, ]: int {0..16}, TENSOR[NPROV, 4])
        lr=1e-2,
        n_iter=2500,
        verbose=False,
    ):
        self.device = device
        vertex_contact = vertex_contact.long()
        anchor_id = anchor_id.long()
        anchor_padding_mask = anchor_padding_mask.long()

        # STEP 0: save all vertices and faces, for collision loss
        self.full_obj_verts_3d = obj_verts_3d
        self.full_obj_faces = obj_faces
        self.full_obj_verts_3d_reduced = obj_verts_3d_reduced
        self.full_obj_faces_reduced = obj_faces_reduced

        # STEP 1: boolean index objvert3d, anchor_id, anchor_elasti && anchor_padding_mask
        self.obj_verts_3d = obj_verts_3d[vertex_contact == 1, :]  # TENSOR[NCONT, 3]
        self.obj_normals = obj_normals[vertex_contact == 1, :]  # TENSOR[NCONT, 3]
        self.obj_contact_region = contact_region[vertex_contact == 1]  # TENSOR[NCONT, ]
        anchor_id = anchor_id[vertex_contact == 1, :]  # TENSOR[NCONT, 4]
        anchor_elasti = anchor_elasti[vertex_contact == 1, :]  # TENSOR[NCONT, 4]
        anchor_padding_mask = anchor_padding_mask[vertex_contact == 1, :]  # TENSOR[NCONT, 4]
        # save hand vertex selection essentials
        self.hand_region_assignment = hand_region_assignment
        self.hand_palm_vertex_mask = hand_palm_vertex_mask

        # STEP 2: boolean mask indexing anchor_id, anchor_elasti && object_vert_id
        self.indexed_anchor_id = anchor_id[anchor_padding_mask == 1]  # TENSOR[NVALID, ]
        self.indexed_anchor_elasti = anchor_elasti[anchor_padding_mask == 1]  # TENSOR[NVALID, ]
        vertex_id = torch.arange(anchor_id.shape[0])[:, None].repeat_interleave(
            anchor_padding_mask.shape[1], dim=1
        )  # TENSOR[NCONT, 4]
        self.indexed_vertex_id = vertex_id[anchor_padding_mask == 1]  # TENSOR[NVALID, ]

        # STEP 3: index anchor && object vert
        self.vertex_pos = self.obj_verts_3d[self.indexed_vertex_id]  # TENSOR[NVALID, 3]

        # STEP 4: prepare essentials for repulsion loss
        obj_vert_idx_list = []
        hand_vert_idx_list = []
        for vertex_id, contact_region_id in enumerate(self.obj_contact_region):
            selected_hand_vert_mask = get_region_palm_mask(
                contact_region_id, None, self.hand_region_assignment, self.hand_palm_vertex_mask
            )
            selected_hand_vert_idx = torch.where(selected_hand_vert_mask)[0]
            repeat_times = selected_hand_vert_idx.shape[0]
            obj_vertex_idx = torch.ones((repeat_times,), dtype=torch.long) * vertex_id
            obj_vert_idx_list.append(obj_vertex_idx)
            hand_vert_idx_list.append(selected_hand_vert_idx)
        self.concat_hand_vert_idx = torch.cat(hand_vert_idx_list, dim=0)
        self.concat_obj_vert_idx = torch.cat(obj_vert_idx_list, dim=0)
        self.concat_obj_vert_3d = self.obj_verts_3d[self.concat_obj_vert_idx, :]
        self.concat_obj_normal = self.obj_normals[self.concat_obj_vert_idx, :]

        # STEP 5: initialize all varaibles
        self.trans_gt = trans_gt
        self.shape_gt = shape_gt
        if pose_gt is None:
            self.sel_pose_idx, self.sel_pose_gt = [], torch.zeros((0, 4), dtype=torch.float).to(self.device)
        else:
            self.sel_pose_idx, self.sel_pose_gt = pose_gt
        self.var_pose_idx = self._get_var_pose_idx(self.sel_pose_idx)
        self.n_var_pose = len(self.var_pose_idx)

        self.vars_hand_trans = None  # TENSOR[3, ]
        self.vars_hand_pose = None  # TENSOR[48, ]
        self.vars_hand_shape = None  # TENSOR[10, ]
        self.optim_vars_list = []
        self.reset_vars()  # vars is intantiated in this function
        self.optim_flag = len(self.optim_vars_list) > 0
        if verbose:
            print("Optimize: ", self.optim_flag)
            print("Optimize: ", self.var_pose_idx)

        # STEP 6: initialize optimizer & torch related stuff
        self.lr = lr
        self.n_iter = n_iter
        if self.optim_flag:
            self.optimizer = torch.optim.Adam(self.optim_vars_list, lr=self.lr,)
        else:
            self.optimizer = None
        self.mano_layer = ManoLayer(
            joint_rot_mode="quat",
            root_rot_mode="quat",
            use_pca=False,
            mano_root="assets/mano",
            center_idx=None,
            flat_hand_mean=True,
        ).to(self.device)
        self.anchor_layer = AnchorLayer("./data/info/anchor").to(self.device)
        self.hand_faces = self.mano_layer.th_faces
        self.static_verts = self.get_static_hand_verts()
        self.hand_edges = self.get_edge_idx(self.hand_faces)
        self.static_edge_len = self.get_edge_len(self.static_verts, self.hand_edges)

        # STEP 7: save loss blend ratio
        # pass

    # todo: add api for initializing pose values, trans values and shape values
    def reset_vars(self):
        self.optim_vars_list = []

        if self.trans_gt is None:
            self.vars_hand_trans = torch.zeros(3, dtype=torch.float, requires_grad=True, device=self.device)
            self.optim_vars_list.append(self.vars_hand_trans)
        else:
            self.vars_hand_trans = self.trans_gt.detach().clone()

        if self.n_var_pose > 0:
            init_val = np.array([[0.9999, 0.0, -0.0101, 0.0]] * self.n_var_pose).astype(np.float32)
            self.vars_hand_pose = torch.tensor(init_val, dtype=torch.float, requires_grad=True, device=self.device)
            self.optim_vars_list.append(self.vars_hand_pose)
        else:
            self.vars_hand_pose = torch.zeros((0, 4), dtype=torch.float, requires_grad=True, device=self.device)

        if self.shape_gt is None:
            self.vars_hand_shape = torch.zeros(10, dtype=torch.float, requires_grad=True, device=self.device)
            self.optim_vars_list.append(self.vars_hand_shape)
        else:
            self.vars_hand_shape = self.shape_gt.detach().clone()

    def get_static_hand_verts(self):
        init_val_pose = np.array([[1.0, 0.0, 0.0, 0.0]] * 16).astype(np.float32)
        vec_pose = torch.tensor(init_val_pose).reshape(-1).unsqueeze(0).float().to(self.device)
        vec_shape = torch.zeros(1, 10).float().to(self.device)
        v, j = self.mano_layer(vec_pose, vec_shape)
        v = v.squeeze(0)
        return v

    # todo: validate correctness of this function
    @staticmethod
    def get_edge_idx(face_idx_tensor: torch.Tensor) -> list:
        device = face_idx_tensor.device
        res = []
        face_idx_tensor = face_idx_tensor.long()
        face_idx_list = face_idx_tensor.tolist()
        for item in face_idx_list:
            v_idx_0, v_idx_1, v_idx_2 = item
            if {v_idx_0, v_idx_1} not in res:
                res.append({v_idx_0, v_idx_1})
            if {v_idx_1, v_idx_2} not in res:
                res.append({v_idx_1, v_idx_2})
            if {v_idx_0, v_idx_2} not in res:
                res.append({v_idx_0, v_idx_2})
        res = [list(e) for e in res]
        res = torch.tensor(res).long().to(device)
        return res

    @staticmethod
    def _get_var_pose_idx(sel_pose_idx):
        # gt has 16 pose
        all_pose_idx = set(range(16))
        sel_pose_idx_set = set(sel_pose_idx)
        var_pose_idx = all_pose_idx.difference(sel_pose_idx_set)
        return list(var_pose_idx)

    @staticmethod
    def assemble_pose_vec(gt_idx, gt_pose, var_idx, var_pose):
        device = var_pose.device
        res = torch.zeros((16, 4), dtype=torch.float).to(device)
        n_gt = len(gt_idx)
        if n_gt > 0:
            gt_pose = gt_pose.reshape((n_gt, 4))
            res[gt_idx, :] = gt_pose
        n_var = len(var_idx)
        if n_var > 0:
            var_pose = var_pose.reshape((n_var, 4))
            res[var_idx, :] = var_pose
        res = res.reshape(-1)
        return res

    @staticmethod
    def contact_loss(apos, vpos, e):
        # apos, vpos = TENSOR[NVALID, 3]
        # e = TENSOR[NVALID, ]
        sqrt_e = torch.sqrt(e)
        dist = torch.sum(torch.pow(vpos - apos, 2), dim=1)  # TENSOR[NVALID, ]
        res = torch.mean(sqrt_e * dist, dim=0)
        return res

    @staticmethod
    def pose_quat_norm_loss(var_pose):
        """ this is the only loss accepts unnormalized quats """
        reshaped_var_pose = var_pose.reshape((16, 4))  # TENSOR[16, 4]
        quat_norm_sq = quaternion_norm_squared(reshaped_var_pose)  # TENSOR[16, ]
        squared_norm_diff = quat_norm_sq - 1.0  # TENSOR[16, ]
        res = torch.mean(torch.pow(squared_norm_diff, 2), dim=0)
        return res

    @staticmethod
    def shape_reg_loss(var_shape):
        return torch.sum(torch.pow(var_shape, 2), dim=0)

    @staticmethod
    def pose_reg_loss(var_pose_normed):
        # the format of quat is [w, x, y, z]
        # to regularize
        # just to make sure w is close to 1.0
        # working aside with self.pose_quat_norm_loss defined above
        reshaped_var_pose_normed = var_pose_normed.reshape((16, 4))
        w = reshaped_var_pose_normed[..., 0]  # get w
        diff = w - 1.0  # TENSOR[16, ]
        res = torch.mean(torch.pow(diff, 2), dim=0)
        return res

    @staticmethod
    def get_edge_len(verts: torch.Tensor, edge_idx: torch.Tensor):
        # verts: TENSOR[NVERT, 3]
        # edge_idx: TENSOR[NEDGE, 2]
        return torch.norm(verts[edge_idx[:, 0], :] - verts[edge_idx[:, 1], :], p=2, dim=1)

    @staticmethod
    def edge_len_loss(rebuild_verts, hand_edges, static_edge_len):
        pred_edge_len = HandOptimizer.get_edge_len(rebuild_verts, hand_edges)
        diff = pred_edge_len - static_edge_len  # TENSOR[NEDGE, ]
        return torch.mean(torch.pow(diff, 2), dim=0)

    @staticmethod
    def joint_smooth_loss(var_pose_normed):
        lvl1_idxs = [1, 4, 7, 10, 13]
        lvl2_idxs = [2, 5, 8, 11, 14]
        lvl3_idxs = [3, 6, 9, 12, 15]
        reshaped_var_pose_normed = var_pose_normed.reshape((16, 4))  # TENSOR[16, 4]
        lvl1_quat = reshaped_var_pose_normed[lvl1_idxs, :]  # TENSOR[5, 4]
        lvl2_quat = reshaped_var_pose_normed[lvl2_idxs, :]  # TENSOR[5, 4]
        lvl3_quat = reshaped_var_pose_normed[lvl3_idxs, :]  # TENSOR[5, 4]
        # first part, lvl2 -> lvl1
        combinated_quat_real_1 = quaternion_mul(quaternion_inv(lvl2_quat), lvl1_quat)[..., 0]  # TENSOR[5, ]
        combinated_quat_real_diff_1 = combinated_quat_real_1 - 1.0
        loss_part_1 = torch.mean(torch.pow(combinated_quat_real_diff_1, 2), dim=0)
        # second part, lvl3 -> lvl2
        combinated_quat_real_2 = quaternion_mul(quaternion_inv(lvl3_quat), lvl2_quat)[..., 0]  # TENSOR[5, ]
        combinated_quat_real_diff_2 = combinated_quat_real_2 - 1.0
        loss_part_2 = torch.mean(torch.pow(combinated_quat_real_diff_2, 2), dim=0)
        # combine them
        res = loss_part_1 + loss_part_2
        return res

    @staticmethod
    def rotation_angle_loss(quat_angle, limit_angle=pi / 2, eps=1e-10):
        # quat_angle: TENSOR[16, ]
        quat_angle_new = torch.zeros_like(quat_angle)  # TENSOR[16, ]
        nonzero_mask = torch.abs(quat_angle) > eps  # TENSOR[16, ], bool
        quat_angle_new[nonzero_mask] = quat_angle[nonzero_mask]  # if angle is too small, pick them out of backward graph
        angle_over_limit = torch.relu(quat_angle_new - pi / 3)  # < pi/2, 0; > pi/2, linear | Tensor[16, ]
        angle_over_limit_squared = torch.pow(angle_over_limit, 2)  # TENSOR[16, ]
        res = torch.mean(angle_over_limit_squared, dim=0)
        return res

    @staticmethod
    def normalize_predaxis(predaxis):
        return F.normalize(predaxis, p=2, dim=1)  # TENSOR[16, 3]

    @staticmethod
    def sik_loss(predaxis_reshaped_normalize):
        """ external implementation """
        ske_index = [2, 5, 8, 11, 14]
        axis_cos = torch.cat(
            [
                torch.abs(
                    torch.sum(
                        predaxis_reshaped_normalize[i, :] * predaxis_reshaped_normalize[i + 1, :], dim=-1, keepdims=True
                    )
                )
                for i in ske_index
            ],
            dim=0,
        )  # TENSOR[5, ]
        roll_over_loss = F.l1_loss(axis_cos, torch.ones_like(axis_cos))
        return roll_over_loss

    @staticmethod
    def inverse_rotation_penalty_loss(predaxis_reshaped_normalize):
        device = predaxis_reshaped_normalize.device
        lvl1_idxs = [1, 4, 7, 10, 14]
        select_axis = predaxis_reshaped_normalize[lvl1_idxs, :]  # TENSOR[4, 3]
        z_axis = torch.tensor([[0.0, 0.0, 1.0]] * len(lvl1_idxs)).to(device)  # TENSOR[4, 3]
        cos_val = select_axis.unsqueeze(1) @ z_axis.unsqueeze(2)  # TENSOR[4, 1, 3] @ TENSOR[4, 3, 1] -> TENSOR[4, 1, 1]
        cos_val = cos_val.squeeze(2).squeeze(1)  # TENSOR[4, ]
        res = F.l1_loss(cos_val, torch.ones_like(cos_val))
        return res

    @staticmethod
    def radial_rotation_penalty_loss(predaxis_reshaped_normalize):
        device = predaxis_reshaped_normalize.device
        lvl1_idxs = [1, 4, 7, 10, 14]
        select_axis = predaxis_reshaped_normalize[lvl1_idxs, :]  # TENSOR[4, 3]
        x_axis = torch.tensor([[1.0, 0.0, 0.0]] * len(lvl1_idxs)).to(device)  # TENSOR[4, 3]
        inner_product = select_axis.unsqueeze(1) @ x_axis.unsqueeze(2)
        # TENSOR[4, 1, 3] @ TENSOR[4, 3, 1] -> TENSOR[4, 1, 1]
        inner_product = inner_product.squeeze(2).squeeze(1)  # TENSOR[4, ]
        res = F.l1_loss(inner_product, torch.zeros_like(inner_product))
        return res

    @staticmethod
    def hand_back_repulsion_direction_loss(
        pred_hand_verts, concat_hand_vert_idx, concat_obj_vert_3d, concat_obj_normal,
    ):
        # pred_hand_verts = TENSOR[NHANDVERTS, 3]
        selected_hand_verts = pred_hand_verts[concat_hand_vert_idx, :]  # TENSOR[NCC, 3]
        # compute offset vector from object to hand
        offset_vectors = selected_hand_verts - concat_obj_vert_3d  # TENSOR[NCC, 3]
        # compute inner product (not normalized)
        inner_product = offset_vectors.unsqueeze(1) @ concat_obj_normal.unsqueeze(2)
        # TENSOR[NCC, 1, 3] @ TENSOR[NCC, 3, 1] -> TENSOR[NCC, 1, 1]
        inner_product = inner_product.squeeze(2).squeeze(1)  # TENSOR[NCC, ]
        thresholded_value = torch.exp(torch.relu(-inner_product)) - 1  # TENSOR[NCC, ]
        # res = torch.mean(torch.pow(thresholded_value, 2), dim=0)
        res = torch.mean(thresholded_value, dim=0)
        return res

    @staticmethod
    def hand_back_repulsion_magnitude_loss(
        pred_hand_verts, concat_hand_vert_idx, concat_obj_vert_3d, constant=0.05, threshold=0.01, eps=1e-10
    ):
        # pred_hand_verts = TENSOR[NHANDVERTS, 3]
        selected_hand_verts = pred_hand_verts[concat_hand_vert_idx, :]  # TENSOR[NCC, 3]
        # compute offset vector from object to hand
        offset_vectors = selected_hand_verts - concat_obj_vert_3d  # TENSOR[NCC, 3]
        # compute norm of vector
        offset = torch.norm(offset_vectors, p=2, dim=1)  # TENSOR[NCC, ]
        # get threshold mask
        n_value = offset.shape[0]
        mask = torch.ones((n_value,), dtype=torch.long)
        mask[offset > threshold] = 0
        # energy value
        energy = constant / (offset + eps)
        # mask and mean
        res = torch.mean(energy * mask)
        return res

    @staticmethod
    def collision_loss_hand_in_obj(
        hand_verts, obj_verts, obj_faces,
    ):
        device = hand_verts.device

        # unsqueeze fist dimension so that we can use hasson's utils directly
        # todo: reimplement this in non batch way
        hand_verts = hand_verts.unsqueeze(0)
        obj_verts = obj_verts.unsqueeze(0)

        # Get obj triangle positions
        obj_triangles = obj_verts[:, obj_faces]
        exterior = batch_mesh_contains_points(
            hand_verts.detach(), obj_triangles.detach()
        )  # exterior computation transfers no gradients
        penetr_mask = ~exterior

        # only compute exterior related stuff
        valid_vals = penetr_mask.sum()
        if valid_vals > 0:
            selected_hand_verts = hand_verts[penetr_mask, :]
            selected_hand_verts = selected_hand_verts.unsqueeze(0)
            dists = batch_pairwise_dist(selected_hand_verts, obj_verts)
            mins_sel_hand_to_obj, mins_sel_hand_to_obj_idx = torch.min(dists, 2)

            results_close = batch_index_select(obj_verts, 1, mins_sel_hand_to_obj_idx)
            collision_vals = ((results_close - selected_hand_verts) ** 2).sum(2)

            penetr_loss = torch.mean(collision_vals)
        else:
            penetr_loss = torch.Tensor([0.0]).to(device)
        return penetr_loss

    def optimize(self, progress=False):
        if progress:
            bar = trange(self.n_iter, position=0)
            bar_hand = trange(0, position=1, bar_format="{desc}")
            bar_contact = trange(0, position=2, bar_format="{desc}")
        else:
            bar = range(self.n_iter)

        loss = torch.Tensor([0.0]).to(self.device)
        for _ in bar:
            if self.optim_flag:
                self.optimizer.zero_grad()

            #  ========== layers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            vars_hand_pose_assembled = self.assemble_pose_vec(
                self.sel_pose_idx, self.sel_pose_gt, self.var_pose_idx, self.vars_hand_pose
            )  # TENSOR[64, ]
            # quat basic normalization loss (soft constraint for unit quat)
            quat_norm_loss = self.pose_quat_norm_loss(vars_hand_pose_assembled)  # TENSOR[]
            # normalize the quat
            vars_hand_pose_normalized = normalize_quaternion(vars_hand_pose_assembled.reshape((16, 4))).reshape(-1)
            # TENSOR[64, ]
            # quat zero rotation loss
            pose_reg_loss = self.pose_reg_loss(vars_hand_pose_normalized)
            # smooth joint
            joint_smooth_loss = self.joint_smooth_loss(vars_hand_pose_normalized)
            # limit rotate direction
            pred_angle_axis = quaternion_to_angle_axis(vars_hand_pose_assembled.reshape((16, 4)))  # TENSOR[16, 3]
            pred_angle_axis_normalized = self.normalize_predaxis(pred_angle_axis)  # TENSOR[16, 3]
            pred_angle = torch.norm(pred_angle_axis, p=2, dim=1)  # TENSOR[16, ]
            roll_over_loss = self.sik_loss(pred_angle_axis_normalized)
            inv_rot_penalty_loss = self.inverse_rotation_penalty_loss(pred_angle_axis_normalized)
            rad_rot_penalty_loss = self.radial_rotation_penalty_loss(pred_angle_axis_normalized)
            # limit angle
            angle_limit_loss = self.rotation_angle_loss(pred_angle)

            # unsqueeze 0 dimension to form a batch with size 1
            vec_pose = vars_hand_pose_assembled.unsqueeze(0)
            vec_shape = self.vars_hand_shape.unsqueeze(0)
            vec_trans = self.vars_hand_trans.unsqueeze(0)
            rebuild_verts, _ = self.mano_layer(vec_pose, vec_shape)
            rebuild_verts = rebuild_verts + vec_trans
            rebuild_anchor = self.anchor_layer(rebuild_verts)
            rebuild_anchor = rebuild_anchor.contiguous()  # TENSOR[1, 32, 3]
            rebuild_anchor = rebuild_anchor.squeeze(0)  # TENSOR[32, 3]

            # index rebuild_anchor
            anchor_pos = rebuild_anchor[self.indexed_anchor_id]  # TENSOR[NVALID, 3]
            contact_loss = self.contact_loss(anchor_pos, self.vertex_pos, self.indexed_anchor_elasti)
            rebuild_verts_squeezed = rebuild_verts.squeeze(0)
            edge_loss = self.edge_len_loss(rebuild_verts_squeezed, self.hand_edges, self.static_edge_len)
            hb_repul_d_loss = self.hand_back_repulsion_direction_loss(
                rebuild_verts_squeezed, self.concat_hand_vert_idx, self.concat_obj_vert_3d, self.concat_obj_normal
            )
            # hb_repul_m_loss = self.hand_back_repulsion_magnitude_loss(
            #     rebuild_verts_squeezed, self.concat_hand_vert_idx, self.concat_obj_vert_3d
            # )
            collision_loss_hio = self.collision_loss_hand_in_obj(
                rebuild_verts_squeezed, self.full_obj_verts_3d_reduced, self.full_obj_faces_reduced
            )

            # get loss
            loss = (
                # HAND SELF LOSS
                1.0 * quat_norm_loss
                + 1.0 * pose_reg_loss
                + 10.0 * joint_smooth_loss
                + 0.1 * angle_limit_loss
                + 10.0 * roll_over_loss
                + 10.0 * edge_loss
                + 1.0 * inv_rot_penalty_loss
                + 0.1 * rad_rot_penalty_loss
                # CONTACT LOSS
                + 1000.0 * contact_loss
                + 100.0 * hb_repul_d_loss
                + 10.0 * collision_loss_hio
            )
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # backward and step
            if self.optim_flag:
                loss.backward()
                self.optimizer.step()

            if progress:
                bar.set_description("TOTAL LOSS {:4e}".format(loss.item()))
                bar_hand.set_description(
                    colored("HAND_REGUL_LOSS: ", "yellow")
                    + "QN={:.3e} PR={:.3e} JS={:3e}, AL={:.3e}, RO={:.3e}, Edge={:.3e}, IR={:.3e}, RR={:.3e}".format(
                        quat_norm_loss.item(),  # QN
                        pose_reg_loss.item(),  # PR
                        joint_smooth_loss.item(),  # JS
                        angle_limit_loss.item(),  # AL
                        roll_over_loss.item(),  # RO
                        edge_loss.item(),  # Edge
                        inv_rot_penalty_loss.item(),  # IR
                        rad_rot_penalty_loss.item(),  # RR
                    )
                )

                bar_contact.set_description(
                    colored("HO_CONTACT_LOSS: ", "blue")
                    + "Conta={:.3e}, Repul={:.3e}, Colli={:.3e}".format(
                        contact_loss.item(),  # Conta
                        hb_repul_d_loss.item(),  # Repul
                        collision_loss_hio.item(),  # Colli
                    )
                )

        return loss.item()

    def recover_hand(self, squeeze_out=True):
        vars_hand_pose_assembled = self.assemble_pose_vec(
            self.sel_pose_idx, self.sel_pose_gt, self.var_pose_idx, self.vars_hand_pose
        ).detach()
        vars_hand_pose_normalized = normalize_quaternion(vars_hand_pose_assembled.reshape((16, 4))).reshape(-1)
        vec_pose = vars_hand_pose_normalized.unsqueeze(0)
        vec_shape = self.vars_hand_shape.detach().unsqueeze(0)
        vec_trans = self.vars_hand_trans.detach().unsqueeze(0)
        rebuild_verts, rebuild_joints = self.mano_layer(vec_pose, vec_shape)
        rebuild_verts = rebuild_verts + vec_trans
        rebuild_joints = rebuild_joints + vec_trans
        if squeeze_out:
            rebuild_verts, rebuild_joints = rebuild_verts.squeeze(0), rebuild_joints.squeeze(0)
        return rebuild_verts, rebuild_joints
