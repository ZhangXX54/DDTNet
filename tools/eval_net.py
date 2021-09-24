import glob
import time
import os
import pickle

import torch
from PIL import Image
from vizer.draw import draw_boxes
from torch.nn import functional as F
from fcos_core.config import cfg
from fcos_core.data.datasets import COCODataset,LYMDataset
import argparse
import numpy as np

from fcos_core.data.transforms import build_transforms
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.miscellaneous import mkdir
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.structures.boxlist_ops import boxlist_ml_nms


maskfcos = True
@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type):
    # if dataset_type == "voc":
    #     class_names = VOCDataset.class_names
    if dataset_type == "lym":
        class_names = ('__background__', 'lym')
    else:
        raise NotImplementedError('Not implemented now.')
    device = torch.device(cfg.MODEL.DEVICE)

    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(cfg.MODEL.WEIGHT)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = glob.glob(os.path.join(images_dir, '*.png'))
    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(is_train=False)
    model.eval()
    all_boxes = [[],[]]
    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)
        image = np.array(Image.open(image_path).convert("RGB"))
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        results = model(images.to(device))
        inference_time = time.time() - start

        boxes = results[0][0].bbox.to(cpu_device).numpy()
        labels = results[0][0].get_field("labels").to(cpu_device).numpy()
        scores = results[0][0].get_field("scores").to(cpu_device).numpy()


        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]

        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))

        drawn_image = draw_boxes(image, boxes, labels, scores, class_names, color=(255, 255, 0), width=3).astype(np.uint8)
        Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

        if maskfcos:
            maskpred = torch.sigmoid(results[1][0])
            maskpred = F.interpolate(maskpred, scale_factor=2)
            maskpred = (maskpred > score_threshold).float().squeeze(0).squeeze(0)
            maskpred = maskpred.data.cpu().numpy().astype(np.uint8)
            Image.fromarray(maskpred).save(os.path.join(output_dir, 'premask_' + image_name))
            contourpred = torch.sigmoid(results[2][0])
            contourpred = F.interpolate(contourpred, scale_factor=2)
            contourpred = (contourpred > score_threshold).float().squeeze(0).squeeze(0)
            contourpred = contourpred.data.cpu().numpy().astype(np.uint8)
            Image.fromarray(contourpred).save(os.path.join(output_dir, 'precontour_' + image_name))
        if len(dets) != 0:
            name = [None for _ in range(len(dets))]
            name[0] = image_path
        else:
            name = [image_path]
            dets = []
        all_boxes[0].append(name)
        all_boxes[1].append(dets)

    all_boxes = np.array(all_boxes)
    with open(os.path.join(output_dir, 'test.txt'), 'wt') as f:
        for k in range(all_boxes.shape[1]):
            if all_boxes[1][k] != []:
                for m in range(len(all_boxes[1][k])):
                    bb = all_boxes[1][k][m]
                    f.write('{:s} {:f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(all_boxes[0][k][0],
                                                                               bb[4],
                                                                               bb[1],
                                                                               bb[0],
                                                                               bb[3],
                                                                               bb[2]))
            else:
                f.write('{:s} {:f} {:f} {:f} {:f} {:f}\n'.format(all_boxes[0][k][0],0,0,0,1,1))
    # np.savetxt(os.path.join(output_dir, 'perdiction_name.txt'), img_name)
    # np.savetxt(os.path.join(output_dir, 'perdiction_baoxes.txt'), all_boxes)

def main():
    parser = argparse.ArgumentParser(description="DSSD Demo.")
    parser.add_argument(
        "--config-file",
        # default= "fcos_test.yaml",
        default="ddtnet_test.yaml",
        # default="retinanet_test.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--images_dir", default='../prediction/test/BCa/320/data1/', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='../prediction/result/BCa/', type=str, help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="lym", type=str, help='Specify dataset type. Currently support voc, coco and lym.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             # dataset=args.dateset,
             dataset_type=args.dataset_type
             )


if __name__ == '__main__':
    main()
