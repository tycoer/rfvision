import numpy as np
import torch


class VertexContactEvalUtil:
    # basically, it is an average meter reimplemented
    # cater to eval util interface

    def __init__(self):
        self._reset()

    def _reset(self):
        self.sum_true_positive = 0
        self.sum_false_positive = 0
        self.sum_true_negative = 0
        self.sum_false_negative = 0
        self.count = 0

        self.acc = 0.0
        self.pc = 0.0
        self.rc = 0.0
        self.f1 = 0.0

    def empty(self):
        return self.count == 0

    def feed(self, vertex_contact_pred_filtered, vertex_contact_gt_filtered):
        batch_size = vertex_contact_pred_filtered.shape[0]
        # compare_result = ((vertex_contact_pred_filtered > 0.5) == vertex_contact_gt_filtered).long()
        # n_correct_pred = torch.sum(compare_result).cpu().item()  # fetch one result back, saving bandwidth
        pred = (vertex_contact_pred_filtered > 0.5).long()
        tp = torch.sum((pred == 1) & (vertex_contact_gt_filtered == 1)).item()
        fp = torch.sum((pred == 1) & (vertex_contact_gt_filtered == 0)).item()
        tn = torch.sum((pred == 0) & (vertex_contact_gt_filtered == 0)).item()
        fn = torch.sum((pred == 0) & (vertex_contact_gt_filtered == 1)).item()
        self.sum_true_positive += tp
        self.sum_false_positive += fp
        self.sum_true_negative += tn
        self.sum_false_negative += fn
        self.count += batch_size

        # ========== calculate >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # accuracy
        self.acc = (self.sum_true_positive + self.sum_true_negative) / self.count

        # precision
        if (self.sum_true_positive + self.sum_false_positive) == 0:
            self.pc = 0.0
        else:
            self.pc = self.sum_true_positive / (self.sum_true_positive + self.sum_false_positive)

        # recall
        if (self.sum_true_positive + self.sum_false_negative) == 0:
            self.rc = 0.0
        else:
            self.rc = self.sum_true_positive / (self.sum_true_positive + self.sum_false_negative)

        # f1 score
        if (2 * self.sum_true_positive + self.sum_false_positive + self.sum_false_negative) == 0:
            self.f1 = 0.0
        else:
            self.f1 = (2 * self.sum_true_positive) / (
                2 * self.sum_true_positive + self.sum_false_positive + self.sum_false_negative
            )
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def get_measures(self):

        return {
            "accuracy": self.acc,
            "precision": self.pc,
            "recall": self.rc,
            "f1_score": self.f1,
        }


class ContactRegionEvalUtil:
    def __init__(self):
        self._reset()

    def _reset(self):
        self.sum = 0
        self.count = 0
        self.acc = 0.0

    def empty(self):
        return self.count == 0

    def feed(self, contact_region_pred_filtered, contact_region_gt_filtered):
        n_points = contact_region_pred_filtered.shape[0]
        n_correct = torch.sum(contact_region_pred_filtered == contact_region_gt_filtered).item()
        self.count += n_points
        self.sum += n_correct
        self.acc = self.sum / self.count

    def get_measures(self):
        return {
            "accuracy": self.acc,
        }
