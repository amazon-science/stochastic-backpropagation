import os
import random
import argparse

import torch

from .config import update_config
from .logger import bootstrap_logger, get_logfile
from .miscs import to_string

LOGGER_SET_FLAG = False


def define_default_arg_parser():
    """Define a default arg_parser.
    Returns: argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        default='',
        type=str,
        help='path to yaml config file',
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs='*',
        help='modify config options using the command-line',
    )
    return parser


def bootstrap(print_cfg=False, default_cfg=None):
    parser = define_default_arg_parser()
    cfg = update_config(parser, default_cfg)

    # Create output dir
    if cfg.SESSION:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.SESSION)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Init ddp
    if hasattr(cfg, 'DDP'):
        cfg.DDP.NUM_GPUS = torch.cuda.device_count()
        if cfg.DDP.MASTER_PORT == '-1':
            cfg.DDP.MASTER_PORT = str(random.randint(10000, 20000))

    logger = setup_logger(cfg)
    if print_cfg:
        logger.info(to_string(cfg))

    return cfg


def setup_logger(cfg):
    """Setup logger.
    Args:
        cfg (dict): The log file will be 'cfg.OUTPUT_DIR/cfg.LOG_NAME'.
    Returns: logger.
    """
    global LOGGER_SET_FLAG
    global logger
    if not LOGGER_SET_FLAG:
        logger = bootstrap_logger(get_logfile(cfg))
        LOGGER_SET_FLAG = True
    else:
        return logger

logger = setup_logger(bootstrap(print_cfg=False))
