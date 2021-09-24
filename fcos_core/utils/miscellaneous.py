# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import os


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def str2bool(s):
    return s.lower() in ('true', '1')