# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .transforms import Compose
# from .transforms import Resize
from .transforms import RandomHorizontalFlip
from .transforms import RandomVerticalFlip
from .transforms import ToTensor
# from .transforms import Normalize
from .transforms import ConvertFromInts
from .transforms import PhotometricDistort

# from .transforms import RandomMirror

from .build import build_transforms
