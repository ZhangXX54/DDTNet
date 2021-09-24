# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from fcos_core.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.mask = cfg.MODEL.DDTNET_ON
    def forward(self, images, targets=None, gt=None, contours=None, center=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)

        if gt is not None:
            boxes, mask, contour, proposal_losses = self.rpn(images, features, targets, gt, contours, center)
        else:
            if self.mask:
                boxes, mask, contour, proposal_losses = self.rpn(images, features, targets)
            else:
                boxes, proposal_losses = self.rpn(images, features, targets)


        if self.roi_heads:
                x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            detector_losses = {}

        if targets is not None:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            if self.training:
                return losses

            if self.mask:
                return boxes, mask, contour, losses
            else:
                return boxes, losses
        else:
            if self.mask:
                return boxes, mask, contour, None
            else:
                return boxes, None