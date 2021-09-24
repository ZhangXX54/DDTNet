"""
This file contains specific functions for computing losses of DDTNet
file
"""

import torch
from torch.nn import functional as F
from torch import nn
import os
# from ..utils import concat_box_prediction_layers
from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist
from torch import Tensor,einsum

INF = 100000000



def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class DDTNETLossComputation(object):
    """
    This class computes the DDTNet losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.DDTNet.LOSS_GAMMA,
            cfg.MODEL.DDTNet.LOSS_ALPHA
        )
        # self.cls_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.fpn_strides = cfg.MODEL.DDTNet.FPN_STRIDES
        self.center_sampling = cfg.MODEL.DDTNet.CENTER_SAMPLING
        self.center_sampling_radius = cfg.MODEL.DDTNet.CENTER_SAMPLING_RADIUS
        self.iou_loss_type = cfg.MODEL.DDTNet.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.DDTNet.NORM_REG_TARGETS

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")


    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets, center):
        object_sizes_of_interest = [
            [-1, 16],
            [16, 32],
            [32, 64],
            [64, 128],
            [128, INF],
        ]

        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        center_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

            for i in range(len(center)):
                level_size = int(num_points_per_level[level] ** 0.5)
                center_per_batch = F.interpolate(center[i].unsqueeze(0) , size=[level_size, level_size])
                if i == 0:
                    center_per_level = center_per_batch
                else:
                    center_per_level = torch.cat([center_per_level,center_per_batch],dim=0)
            center_level_first.append(center_per_level)
        return labels_level_first, reg_targets_level_first, center_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def mask_loss(self, mask, gt, contour, contours):
        criterion = nn.BCEWithLogitsLoss()
        size = gt[0].shape
        mask160 = F.interpolate(mask[0], size=[size[1], size[2]])
        if len(gt) > 2:
            GT = torch.stack((gt[0], gt[1]), dim=0)
            for i in range(len(gt)-2):
                GT = torch.cat((GT,gt[i+2].unsqueeze(0)), dim=0)
        elif len(gt) == 2:
            GT = torch.stack((gt[0], gt[1]), dim=0)
        else:
            GT = gt[0].unsqueeze(0)
        loss1 = criterion(mask160, GT)

        if contour is not None:
            contour160 = F.interpolate(contour[0], size=[size[1], size[2]])
            if len(contours) > 2:
                GT = torch.stack((contours[0], contours[1]), dim=0)
                for i in range(len(contours)-2):
                    GT = torch.cat((GT,contours[i+2].unsqueeze(0)), dim=0)
            elif len(contours) == 2:
                GT = torch.stack((contours[0], contours[1]), dim=0)
            else:
                GT = contours[0].unsqueeze(0)
            loss2 = criterion(contour160, GT)
        else:
            loss2 = 0
        return loss1 + loss2


    def __call__(self, locations, box_cls, box_regression, centerness, targets, mask, gt=None, contour=None, contours=None, center=None):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])
            gt (list[Tensor])
            contour (list[Tensor])
            contours (list[Tensor])
            center (list[Tensor])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
            mask_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        labels, reg_targets, center = self.prepare_targets(locations, targets, center)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        center_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))
            center_flatten.append(center[l].reshape(-1))


        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        center_flatten = torch.cat(center_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]


        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / num_pos_avg_per_gpu

        if mask is not None:
            mask_loss = self.mask_loss(
                mask,
                gt,
                contour,
                contours
            )
        else:
            mask_loss = 0

        if pos_inds.numel() > 0:
            if "original" in self.center_sampling:
                # centerness_flatten = centerness_flatten[pos_inds]
                centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
                sum_centerness_targets_avg_per_gpu = \
                    reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

                reg_loss = self.box_reg_loss_func(
                    box_regression_flatten,
                    reg_targets_flatten,
                    centerness_targets
                ) / sum_centerness_targets_avg_per_gpu
                centerness_loss = self.centerness_loss_func(
                    centerness_flatten,
                    centerness_targets
                ) / num_pos_avg_per_gpu
            if "manual dots" in self.center_sampling:
                centerness_weith = center_flatten[pos_inds]
                reg_loss = self.box_reg_loss_func(
                    box_regression_flatten,
                    reg_targets_flatten,
                    centerness_weith
                ) / num_pos_avg_per_gpu
                centerness_loss = self.centerness_loss_func(
                    centerness_flatten,
                    centerness_weith
                ) / num_pos_avg_per_gpu

        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss, mask_loss


def make_ddtnet_loss_evaluator(cfg):
    loss_evaluator = DDTNETLossComputation(cfg)
    return loss_evaluator
