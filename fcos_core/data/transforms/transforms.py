# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import cv2
import torch
import torchvision
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None, gt=None, contour=None, centerness=None):
        for t in self.transforms:
            image, target, gt, contour, centerness = t(image, target, gt, contour, centerness)
        return image, target, gt, contour, centerness

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ConvertFromInts(object):
    def __call__(self, image, target=None, gt=None, contour=None, centerness=None):
        return image.astype(np.float32), target, gt.astype(np.float32), contour.astype(np.float32), centerness.astype(np.float32)

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, target=None, gt=None, contour=None, centerness=None):
        if random.randint(0,2):
            swap = self.perms[random.randint(0,len(self.perms)-1)]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, target, gt, contour, centerness

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, target=None, gt=None, contour=None, centerness=None):
        if random.randint(0,2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, target, gt, contour, centerness

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target, gt, contour, centerness):
        if random.random() < self.prob:
            image = Image.fromarray(image.astype(np.uint8))
            image = F.hflip(image)
            gt = Image.fromarray(gt.repeat(3,axis=2).astype(np.uint8))
            gt = F.hflip(gt)
            target = target.transpose(0)
            gt = np.array(gt)[:,:, 1]

            contour = Image.fromarray(contour.repeat(3,axis=2).astype(np.uint8))
            contour = F.hflip(contour)
            contour = np.array(contour)[:,:, 1]
            centerness = Image.fromarray(centerness.repeat(3,axis=2).astype(np.uint8))
            centerness = F.hflip(centerness)
            centerness = np.array(centerness)[:,:, 1]
            return np.array(image), target, np.expand_dims(gt, axis=2), np.expand_dims(contour, axis=2), np.expand_dims(centerness, axis=2)
        else:
            return np.array(image), target, np.array(gt), np.array(contour), np.array(centerness)

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target, gt, contour, centerness):
        if random.random() < self.prob:
            image = Image.fromarray(image.astype(np.uint8))
            image = F.vflip(image)
            gt = Image.fromarray(gt.repeat(3,axis=2).astype(np.uint8))
            gt = F.vflip(gt)
            target = target.transpose(1)
            gt = np.array(gt)[:,:, 1]

            contour = Image.fromarray(contour.repeat(3,axis=2).astype(np.uint8))
            contour = F.vflip(contour)
            contour = np.array(contour)[:,:, 1]
            centerness = Image.fromarray(centerness.repeat(3,axis=2).astype(np.uint8))
            centerness = F.vflip(centerness)
            centerness = np.array(centerness)[:,:, 1]
            return np.array(image), target, np.expand_dims(gt, axis=2), np.expand_dims(contour, axis=2), np.expand_dims(centerness, axis=2)
        else:
            return np.array(image), target, np.array(gt), np.array(contour), np.array(centerness)


class ToTensor(object):
    def __call__(self, image, target=None, gt=None, contour=None, centerness=None):
        if gt is not None:
            return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1), target, torch.from_numpy(gt.astype(np.float32)).permute(2, 0, 1), torch.from_numpy(contour.astype(np.float32)).permute(2, 0, 1), torch.from_numpy(centerness.astype(np.float32)).permute(2, 0, 1)
        else:
            return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1), target, gt, contour, centerness

class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None, gt=None, contour=None, centerness=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target, gt, contour, centerness


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target=None, gt=None, contour=None, centerness=None):
        if random.randint(0,2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, target, gt, contour, centerness
class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, target=None, gt=None, contour=None, centerness=None):
        if random.randint(0,2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, target, gt, contour, centerness
class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, target=None, gt=None, contour=None, centerness=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, target, gt, contour, centerness
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, target=None, gt=None, contour=None, centerness=None):
        if random.randint(0,2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, target, gt, contour, centerness
class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, target=None, gt=None, contour=None, centerness=None):
        if random.randint(0, 2):
            im = image.copy()
            im, target, gt, contour, centerness = self.rand_brightness(im, target, gt, contour, centerness)
            if random.randint(0,2):
                if random.randint(0, 2):
                    distort = Compose(self.pd[:-1])
                else:
                    distort = Compose(self.pd[1:])
                im, target, gt, contour, centerness = distort(im, target, gt, contour, centerness)

            if random.randint(0, 2):
                im, target, gt, contour, centerness = self.rand_light_noise(im, target, gt, contour, centerness)
            return im, target, gt, contour, centerness
        else:
            return image, target, gt, contour, centerness