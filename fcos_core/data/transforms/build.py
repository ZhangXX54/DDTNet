# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .transforms import *
from . import transforms as T


def build_transforms(is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
        ]

    else:
        transform = [
            ToTensor()
        ]
    transform = T.Compose(transform)
    return transform
