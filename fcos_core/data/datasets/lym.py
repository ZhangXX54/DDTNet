import os

import torch
import torch.utils.data
from PIL import Image
import numpy as np
import sys
import cv2
from torchvision.transforms import functional as F

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from fcos_core.structures.bounding_box import BoxList



class LYMDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "lym",
    )

    def __init__(self, data_dir, data_name, split, use_difficult=False,transforms=None):
        self.data_dir = data_dir
        self.data_name = data_name
        self.split = split
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms
        image_sets_file = os.path.join(self.data_dir, "ImageSets/%s/%s.txt" % (self.data_name, self.split))
        self.ids = LYMDataset._read_image_ids(image_sets_file)


        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = LYMDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = self._read_image(img_id)
        contour = self._read_contour(img_id)
        centerness = self._read_centerness(img_id)
        mask, target = self.get_groundtruth(img_id)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target, mask, contour, centerness = self.transforms(img, target, mask, contour, centerness)

        return img, target, mask, contour, index, centerness


    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "Images/320", "%s.png" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

    def _read_contour(self, image_id):
        contour_file = os.path.join(self.data_dir, 'Masks/320','edgemask_' + "%s.png" % image_id)
        contour = cv2.imread(contour_file, cv2.IMREAD_GRAYSCALE)
        contour = np.array(contour, np.float32)
        contour = np.expand_dims(contour, axis=2)
        contour = contour/255.0
        return contour

    def _read_centerness(self, image_id):
        centerness_file = os.path.join(self.data_dir, 'Masks/320','Dis_' + "%s.png" % image_id)
        centerness = cv2.imread(centerness_file, cv2.IMREAD_GRAYSCALE)
        centerness = np.array(centerness, np.float32)
        centerness = np.expand_dims(centerness, axis=2)
        return centerness

    def get_groundtruth(self, image_id):
        mask, anno = self._preprocess_annotation(image_id)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        return mask, target

    def _preprocess_annotation(self, img_id):
        anno_file = os.path.join(self.data_dir, 'Masks/320','mask_' + "%s.png" % img_id)
        mask = cv2.imread(anno_file, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask>0, 1, mask)
        mask = np.array(mask, np.float32)
        mask = np.expand_dims(mask, axis=2)


        bboxes = []
        labels = []
        mask_gt = cv2.imread(anno_file)
        h, w, _ = mask_gt.shape
        cond1 = mask_gt[:, :, 0] != mask_gt[:, :, 1]
        cond2 = mask_gt[:, :, 1] != mask_gt[:, :, 2]
        cond3 = mask_gt[:, :, 2] != mask_gt[:, :, 0]

        r, c = np.where(np.logical_or(np.logical_or(cond1, cond2), cond3))
        if len(r):
            unique_colors = np.unique(mask_gt[r, c, :], axis=0)

            for color in unique_colors:
                cond1 = mask_gt[:, :, 0] == color[0]
                cond2 = mask_gt[:, :, 1] == color[1]
                cond3 = mask_gt[:, :, 2] == color[2]
                r, c = np.where(np.logical_and(np.logical_and(cond1, cond2), cond3))
                y1 = np.min(r)
                x1 = np.min(c)
                y2 = np.max(r)
                x2 = np.max(c)
                if (abs(y2 - y1) <= 1 or abs(x2 - x1) <= 1):
                    continue
                bboxes.append([x1, y1, x2, y2])  # 512 x 640
                labels.append(1)
            if len(bboxes) == 0:
                bboxes.append([0, 0, 2, 2])
                labels.append(0)
        if len(r) == 0:
            bboxes.append([0, 0, 2, 2])
            labels.append(0)

        im_info = tuple(map(int, (h, w)))
        res = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels),
            "im_info": im_info,
        }
        return mask, res

    def get_img_info(self, index):
        img_id = self.ids[index]
        image_file = os.path.join(self.data_dir, "Images/320", "%s.png" % img_id)
        img = cv2.imread(image_file)
        h, w, _ = img.shape
        im_info = tuple(map(int, (h, w)))
        return {"height": im_info[0], "width": im_info[1], "id":img_id}

    def map_class_id_to_class_name(self, class_id):
        return LYMDataset.CLASSES[class_id]


class LYMTestDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "lym",
    )

    def __init__(self, data_dir, data_name, split, use_difficult=False,transforms=None):
        self.data_dir = data_dir
        self.data_name = data_name
        self.split = split
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms
        image_sets_file = os.path.join(self.data_dir, "ImageSets/%s/%s.txt" % (self.data_name, self.split))
        self.ids = LYMTestDataset._read_image_ids(image_sets_file)


        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = LYMTestDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, index):

        img_id = self.ids[index]
        img = self._read_image(img_id)
        contour = self._read_contour(img_id)
        centerness = self._read_centerness(img_id)
        mask = self._read_gt(img_id)
        target = self.get_groundtruth(img_id)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target, mask, contour, centerness = self.transforms(img, target, mask, contour, centerness)

        return img, target, mask, contour, index, centerness



    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "Images/320", "%s.png" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

    def _read_gt(self, image_id):
        gt_file = os.path.join(self.data_dir, 'Masks/320','mask_' + "%s.png" % image_id)
        gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        gt = np.where(gt > 0, 1, gt)
        gt = np.array(gt, np.float32)
        gt = np.expand_dims(gt, axis=2)
        return gt

    def _read_contour(self, image_id):
        contour_file = os.path.join(self.data_dir, 'Masks/320','edgemask_' + "%s.png" % image_id)
        contour = cv2.imread(contour_file, cv2.IMREAD_GRAYSCALE)
        contour = np.array(contour, np.float32)
        contour = np.expand_dims(contour, axis=2)
        contour = contour/255.0
        return contour

    def _read_centerness(self, image_id):
        centerness_file = os.path.join(self.data_dir, 'Masks/320','Dis_' + "%s.png" % image_id)
        centerness = cv2.imread(centerness_file, cv2.IMREAD_GRAYSCALE)
        centerness = np.array(centerness, np.float32)
        centerness = np.expand_dims(centerness, axis=2)
        return centerness

    def get_groundtruth(self, image_id):
        anno = self._preprocess_annotation(image_id)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        # target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, img_id):
        anno_file = os.path.join(self.data_dir, 'Masks/320','mask_' + "%s.png" % img_id)
        bboxes = []
        labels = []
        mask_gt = cv2.imread(anno_file)
        h, w, _ = mask_gt.shape
        cond1 = mask_gt[:, :, 0] != mask_gt[:, :, 1]
        cond2 = mask_gt[:, :, 1] != mask_gt[:, :, 2]
        cond3 = mask_gt[:, :, 2] != mask_gt[:, :, 0]

        r, c = np.where(np.logical_or(np.logical_or(cond1, cond2), cond3))
        if len(r):
            unique_colors = np.unique(mask_gt[r, c, :], axis=0)

            for color in unique_colors:
                cond1 = mask_gt[:, :, 0] == color[0]
                cond2 = mask_gt[:, :, 1] == color[1]
                cond3 = mask_gt[:, :, 2] == color[2]
                r, c = np.where(np.logical_and(np.logical_and(cond1, cond2), cond3))
                y1 = np.min(r)
                x1 = np.min(c)
                y2 = np.max(r)
                x2 = np.max(c)
                if (abs(y2 - y1) <= 1 or abs(x2 - x1) <= 1):
                    continue
                bboxes.append([x1, y1, x2, y2])  # 512 x 640
                labels.append(1)
            if len(bboxes) == 0:
                bboxes.append([0, 0, 2, 2])
                labels.append(0)
        if len(r) == 0:
            bboxes.append([0, 0, 2, 2])
            labels.append(0)

        im_info = tuple(map(int, (h, w)))
        res = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        image_file = os.path.join(self.data_dir, "Images/320", "%s.png" % img_id)
        img = cv2.imread(image_file)
        h, w, _ = img.shape
        im_info = tuple(map(int, (h, w)))
        return {"height": im_info[0], "width": im_info[1], "id":img_id}

    def map_class_id_to_class_name(self, class_id):
        return LYMTestDataset.CLASSES[class_id]
