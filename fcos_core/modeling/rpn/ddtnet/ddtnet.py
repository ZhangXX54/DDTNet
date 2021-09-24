import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_ddtnet_loss_evaluator

from fcos_core.layers import Scale
from fcos_core.layers import DFConv2d

class DDTNETHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DDTNETHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.DDTNet.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.DDTNet.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.DDTNet.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.DDTNet.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.DDTNet.USE_DCN_IN_TOWER

        cls_tower = []
        bbox_tower = []
        mask_tower = []
        contour_tower = []
        for i in range(cfg.MODEL.DDTNet.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.DDTNet.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())



        for i in range(cfg.MODEL.DDTNet.NUM_CONVS):
            mask_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            mask_tower.append(nn.GroupNorm(32, in_channels))
            mask_tower.append(nn.ReLU())

            contour_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            contour_tower.append(nn.GroupNorm(32, in_channels))
            contour_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.add_module('mask_tower', nn.Sequential(*mask_tower))
        self.add_module('contour_tower', nn.Sequential(*contour_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        self.maskchannel = nn.Conv2d(
            in_channels * 5, 5, kernel_size=1, stride=1,
        )
        self.contourchannel = nn.Conv2d(
            in_channels * 5, 5, kernel_size=1, stride=1,
        )
        self.mask = nn.Conv2d(
            5, 1, kernel_size=3, stride=1,
            padding=1
        )

        self.contour = nn.Conv2d(
            5, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,self.mask_tower, self.contour_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness, self.mask, self.contour, self.maskchannel, self.contourchannel]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.DDTNet.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(6)])



    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        mask = []
        contour = []
        for l, feature in enumerate(x):
            if l == 0 :
                mask_tower = self.mask_tower(feature)
                contour_tower = self.contour_tower(feature)


            if l == 2 or l == 3 or l == 4 or l == 5 or l == 1:
                cls_tower = self.cls_tower(feature)
                box_tower = self.bbox_tower(feature)

                logits.append(self.cls_logits(cls_tower))
                if self.centerness_on_reg:
                    centerness.append(self.centerness(box_tower))
                else:
                    centerness.append(self.centerness(cls_tower))

                # connect
                if l == 1:
                    mask80 = nn.UpsamplingNearest2d(scale_factor=2)(cls_tower)
                    contour80 = nn.UpsamplingNearest2d(scale_factor=2)(box_tower)
                if l == 2:
                    mask40 = nn.UpsamplingNearest2d(scale_factor=4)(cls_tower)
                    contour40 = nn.UpsamplingNearest2d(scale_factor=4)(box_tower)
                if l == 3:
                    mask20 = nn.UpsamplingNearest2d(scale_factor=8)(cls_tower)
                    contour20 = nn.UpsamplingNearest2d(scale_factor=8)(box_tower)
                if l == 4:
                    mask10 = nn.UpsamplingNearest2d(scale_factor=16)(cls_tower)
                    contour10 = nn.UpsamplingNearest2d(scale_factor=16)(box_tower)



                bbox_pred = self.scales[l-1](self.bbox_pred(box_tower))
                if self.norm_reg_targets:
                    bbox_pred = F.relu(bbox_pred)
                    if self.training:
                        bbox_reg.append(bbox_pred)
                    else:
                        bbox_reg.append(bbox_pred * self.fpn_strides[l])
                else:
                    bbox_reg.append(torch.exp(bbox_pred))





        mask.append(self.mask(self.maskchannel(torch.cat((mask_tower, mask80, mask40, mask20, mask10),dim=1))))
        contour.append(self.contour(self.contourchannel(torch.cat((contour_tower, contour80, contour40, contour20, contour10), dim=1))))


        return logits, bbox_reg, centerness, mask, contour


class DDTNETModule(torch.nn.Module):
    """
    Module for DDTNet computation. Takes feature maps from the backbone and
    DDTNet outputs and losses.
    """

    def __init__(self, cfg, in_channels):
        super(DDTNETModule, self).__init__()

        head = DDTNETHead(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_ddtnet_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.DDTNet.FPN_STRIDES

    def forward(self, images, features, targets=None, gt=None, contours=None, center=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            masks (dict[Tensor]): the predicted masks and contours from P1 of RPN.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness, mask, contour = self.head(features)
        locations = self.compute_locations(features)

        if self.training:
            return self._forward_train(
                locations, box_cls,
                box_regression,
                centerness, targets, mask, gt, contour, contours, center

            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression,
                centerness, images.image_sizes, targets, mask, gt, contour, contours, center
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets, mask, gt, contour, contours, center):
        loss_box_cls, loss_box_reg, loss_centerness, loss_mask = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets, mask, gt, contour, contours, center
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness,
            "loss_mask": loss_mask,
        }
        return None, None, None, losses


    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes, targets, mask, gt, contour, contours, center):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression,
            centerness, image_sizes
        )


        if targets is not None:
            loss_box_cls, loss_box_reg, loss_centerness, loss_mask = self.loss_evaluator(
                locations, box_cls, box_regression, centerness, targets, mask, gt, contour, contours, center
            )
            losses = {
                "loss_cls": loss_box_cls,
                "loss_reg": loss_box_reg,
                "loss_centerness": loss_centerness,
                "loss_mask": loss_mask,
            }
            return boxes, mask, contour,losses
        else:
            return boxes, mask, contour,None

    def compute_locations(self, features):
        locations = []
        for l, feature in enumerate(features):
            if l == 2 or l == 3 or l == 4 or l == 5 or l == 1:
                h, w = feature.size()[-2:]
                locations_per_level = self.compute_locations_per_level(
                    h, w, self.fpn_strides[l-1],
                    feature.device
                )
                locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


def build_ddtnet(cfg, in_channels):
    return DDTNETModule(cfg, in_channels)
