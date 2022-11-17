import torch

from dlhammer.checkpointer import Checkpointer
from dlhammer.env import setup_random_seed
from dlhammer.nested import nested_call
from lstr.builder import build_models
from lstr.trainer.inferencer.e2e_perframe_det_batch_inference import \
    do_e2e_perframe_det_batch_inference


def evaluate(cfg, logger):
    # Setup configurations
    setup_random_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpointer = Checkpointer(cfg, phase="train")

    # Build model
    models = build_models(cfg, device, -1)
    nested_call(models, "eval")
    # Load pretrained model and optimizer
    checkpointer.load(models)

    do_e2e_perframe_det_batch_inference(cfg, models, device, logger)
