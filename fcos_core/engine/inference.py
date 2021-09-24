# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm
from fcos_core.data import make_data_loader
from fcos_core.config import cfg
from fcos_core.data.datasets.evaluation import evaluate
from fcos_core.utils.metric_logger import MetricLogger
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
from fcos_core.utils.miscellaneous import mkdir

def compute_on_dataset(cfg, model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    loss = MetricLogger(delimiter="  ")
    cpu_device = torch.device("cpu")
    for iter, batch in enumerate(tqdm(data_loader)):
        images, targets, masks, contours, image_id, centers = batch
        target = [target.to(device) for target in targets]
        mask = [mask.to(device) for mask in masks]
        contour = [contour.to(device) for contour in contours]
        center = [center.to(device) for center in centers]
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            if cfg.MODEL.DDTNET_ON :
                output, mask, contour, losses = model(images.to(device), target, mask, contour, center)
                vallosses = sum(loss for loss in losses.values())
            else:
                output, losses = model(images.to(device), target)
                vallosses = sum(loss for loss in losses.values())
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            bbox = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_id, bbox)}
        )
        loss.update(loss=vallosses, **losses)
    return results_dict, loss


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu[0])
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("fcos_core.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions, predictions_per_gpu[1]


def inference(
        model,
        data_loader,
        dataset_name,
        # iou_types=("bbox",),
        # box_only=False,
        device="cuda",
        # expected_results=(),
        # expected_results_sigma_tol=4,
        ovthresh=0.5,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    # device = torch.device(device)
    # num_devices = get_world_size()
    logger = logging.getLogger("fcos_core.inference")
    dataset = data_loader.dataset

    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(cfg, model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    # total_time = total_timer.toc()
    # total_time_str = get_time_str(total_time)
    # logger.info(
    #     "Total run time: {} ({} s / img per device, on {} devices)".format(
    #         total_time_str, total_time * num_devices / len(dataset), num_devices
    #     )
    # )
    # total_infer_time = get_time_str(inference_timer.total_time)
    # logger.info(
    #     "Model inference time: {} ({} s / img per device, on {} devices)".format(
    #         total_infer_time,
    #         inference_timer.total_time * num_devices / len(dataset),
    #         num_devices,
    #     )
    # )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    # if output_folder:
    #     torch.save(predictions[0], os.path.join(output_folder, "predictions.pth"))

    # extra_args = dict(
    #     box_only=box_only,
    #     iou_types=iou_types,
    #     expected_results=expected_results,
    #     expected_results_sigma_tol=expected_results_sigma_tol,
    # )

    return evaluate(
                    dataset,
                    predictions,
                    output_folder,
                    ovthresh,
                    )

# def inference(model, data_loader, dataset_name, device, ovthresh,output_folder=None, use_cached=False,**kwargs):
#     dataset = data_loader.dataset
#     logger = logging.getLogger("FCOS.inference")
#     logger.info("Evaluating {} dataset({} images):".format(dataset_name, len(dataset)))
#     predictions_path = os.path.join(output_folder, 'predictions.pth')
#     if use_cached and os.path.exists(predictions_path):
#         predictions = torch.load(predictions_path, map_location='cpu')
#     else:
#         predictions = compute_on_dataset(model, data_loader, device)
#         synchronize()
#         predictions = _accumulate_predictions_from_multiple_gpus(predictions)
#     if not is_main_process():
#         return
#     if output_folder:
#         torch.save(predictions, predictions_path)
#     return evaluate(dataset=dataset, predictions=predictions, output_folder=output_folder, ovthresh=ovthresh,**kwargs)

@torch.no_grad()
def run_test(cfg, model, distributed, ovthresh):
    if distributed:
        model = model.module
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    eval_results = []
    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        if not os.path.exists(output_folder):
            mkdir(output_folder)
        eval_result=inference(
                                model,
                                data_loader,
                                dataset_name=dataset_name,
                                # iou_types=iou_types,
                                # box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                                device=device,
                                # expected_results=cfg.TEST.EXPECTED_RESULTS,
                                # expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                                ovthresh=ovthresh,
                                output_folder=output_folder,
                            )
        # synchronize()
        eval_results.append(eval_result)
    return eval_results