import logging
import os
from datetime import datetime

import numpy as np

from .lym_eval import do_lym_evaluation

def lym_evaluation(dataset, predictions, output_folder, ovthresh):
    logger = logging.getLogger("DDTNet.inference")
    logger.info("performing lym evaluation, ignored iou_types.")
    return do_lym_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        ovthresh=ovthresh,
    )