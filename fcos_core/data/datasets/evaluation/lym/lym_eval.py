from __future__ import division
import os
import logging
from collections import defaultdict
from datetime import datetime
import itertools
import numpy as np
import six
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import boxlist_iou

def do_lym_evaluation(dataset, predictions, output_folder, ovthresh):
    # for the user to choose
    pred_boxlists = []
    gt_boxlists = []
    for image_id, prediction in enumerate(predictions[0]):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_groundtruth(img_info['id'])
        gt_boxlists.append(gt_boxlist)
    result = eval_detection_lym(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=ovthresh,
    )
    logger = logging.getLogger("DDTNet.inference")
    result_str = "f1: {:.4f}\n".format(result["f1"])
    metrics1 = {'prec': result["prec"], 'rec': result["rec"], 'f1': result["f1"]}
    logger.info(result_str)


    losses = predictions[1]
    loss = losses.loss.avg

    #DDTNet
    loss_cls = losses.loss_cls.avg
    loss_reg = losses.loss_reg.avg
    loss_centerness = losses.loss_centerness.avg
    loss_mask = losses.loss_mask.avg
    loss_str = "loss: {:.4f},loss_cls: {:.4f},loss_reg: {:.4f},loss_centerness: {:.4f},loss_mask: {:.4f}\n".format(loss, loss_cls,loss_reg,loss_centerness,loss_mask)
    metrics2 = {'loss': loss, 'loss_cls': loss_cls, 'loss_reg': loss_reg, 'loss_centerness':loss_centerness, 'loss_mask': loss_mask}


    logger.info(loss_str)

    return dict(metrics1=metrics1,metrics2=metrics2)




def eval_detection_lym(pred_boxlists, gt_boxlists, iou_thresh):
    """Evaluate on lym dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec, f1 = calc_detection_lym_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh, gt_difficults=None
    )

    return {"prec": np.nanmean(prec[1]), "rec": np.nanmean(rec[1]), "f1": np.nanmean(f1[1])}


def calc_detection_lym_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5, gt_difficults=None):
    """Calculate precision, recall and f1 based on evaluation code of PASCAL VOC.
    This function calculates precision, recall and f1 of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        if gt_difficults is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)
        else:
            gt_difficult = gt_boxlist.get_field("difficult").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    # if gt_difficult_l[gt_idx]:
                    #     match[l].append(-1)

                    if not selec[gt_idx]:
                        match[l].append(1)
                    else:
                        match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class
    f1 = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
        f1[l] = 2 * prec[l] * rec[l] / (prec[l] + rec[l])

    return prec, rec, f1



