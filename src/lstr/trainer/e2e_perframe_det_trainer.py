import time

import torch
import torch.nn as nn
from easydict import EasyDict
from tqdm import tqdm

from dlhammer import distributed as du

from ..evaluation import compute_result
from ..utils.model_util import freeze_bn_statistics
from .partial_feedback import partial_feedback_forward, vanilla_forward


def do_e2e_perframe_det_train(
    cfg,
    data_loaders,
    models,
    criterion,
    optimizers,
    schedulers,
    device,
    checkpointer,
    logger,
):
    # Setup model on multiple GPUs
    ddp_enabled = cfg.DDP.ENABLE
    do_logging = (not ddp_enabled) or (ddp_enabled and cfg.DDP.LOCAL_RANK == 0)
    if not ddp_enabled and torch.cuda.device_count() > 1:
        models = EasyDict({k: nn.DataParallel(v) for k, v in models.items()})

    def train_step(inp, tar):
        if cfg.PARTIAL_FEEDBACK.ENABLE:
            det_logits = partial_feedback_forward(
                models, inp, sample_rate=cfg.PARTIAL_FEEDBACK.SAMPLE_RATE
            )
        else:
            if cfg.MODEL.SPATIAL.FREEZE_ALL:
                split_size = int(inp[0].shape[1])  # T
            else:
                split_size = -1
            det_logits = vanilla_forward(
                models,
                inp,
                freeze_backbone=cfg.MODEL.SPATIAL.FREEZE_ALL,
                split_size=split_size,
            )
        det_tar = tar.reshape(-1, cfg.DATA.NUM_CLASSES)
        det_logits = det_logits.reshape(-1, cfg.DATA.NUM_CLASSES)  # (BxT, NUM_CLASSES)
        det_loss = criterion["MCE"](det_logits, det_tar)

        [optim.zero_grad() for optim in optimizers.values()]
        det_loss.backward()
        [
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            for model in models.values()
        ]
        [optim.step() for optim in optimizers.values()]
        [scheduler.step() for scheduler in schedulers.values()]
        return det_loss, (det_logits, det_tar)

    def test_step(inp, tar):
        split_size = int(inp[0].shape[1])  # T
        det_logits = vanilla_forward(models, inp, split_size=split_size)
        det_tar = tar.reshape(-1, cfg.DATA.NUM_CLASSES)
        det_logits = det_logits.reshape(-1, cfg.DATA.NUM_CLASSES)  # (BxT, NUM_CLASSES)
        det_loss = criterion["MCE"](det_logits, det_tar)
        return det_loss, (det_logits, det_tar)

    step_fn = {"train": train_step, "test": test_step}

    for epoch in range(cfg.SOLVER.START_EPOCH, 1 + cfg.SOLVER.NUM_EPOCHS):
        # Reset
        det_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        det_pred_scores = []
        det_gt_targets = []

        start = time.time()
        for phase in cfg.SOLVER.PHASES:
            training = phase == "train"
            data_loaders[phase].sampler.set_epoch(epoch)

            [model.train(training) for model in models.values()]
            ## freeze bn
            if cfg.MODEL.SPATIAL.FREEZE_BN:
                freeze_bn_statistics(models.spatial_model)

            with torch.set_grad_enabled(training):
                if do_logging:
                    pbar = tqdm(
                        data_loaders[phase],
                        desc="{}ing epoch {}".format(phase.capitalize(), epoch),
                    )
                else:
                    pbar = data_loaders[phase]
                for batch_idx, (inp, tar) in enumerate(pbar, start=1):
                    batch_size = tar.shape[0]

                    if cfg.INPUT.MODALITY == "visual":
                        inp[1] = None
                    elif cfg.INPUT.MODALITY == "motion":
                        inp[0] = None

                    # to gpu
                    inp = [x.to(device) if x is not None else x for x in inp]
                    tar = tar.to(device)

                    # step
                    det_loss, (det_logits, det_target) = step_fn[phase](inp, tar)
                    if ddp_enabled:
                        [det_loss] = du.all_reduce([det_loss])
                        [det_logits, det_target] = du.all_gather(
                            [det_logits, det_target]
                        )

                    # log metrics
                    det_losses[phase] += (
                        det_loss.item() * batch_size * du.get_world_size()
                    )

                    # Output log for current batch
                    if do_logging:
                        show = {
                            key + ".lr": f"{scheduler.get_last_lr()[0]:.7f}"
                            for key, scheduler in schedulers.items()
                        }
                        show["det_loss"] = f"{det_loss:.5f}"
                        pbar.set_postfix(show)

                    if not training:
                        # Prepare for evaluation
                        det_score = (
                            det_logits.softmax(dim=1).cpu().tolist()
                        )  # shape: (BxT,C)
                        det_target = det_target.cpu().tolist()
                        det_pred_scores.extend(det_score)
                        det_gt_targets.extend(det_target)
        end = time.time()
        du.synchronize()

        # Output log for current epoch
        log = []
        log.append("Epoch {:2}".format(epoch))
        log.append(
            "train det_loss: {:.5f}".format(
                det_losses["train"] / len(data_loaders["train"].dataset),
            )
        )
        if "test" in cfg.SOLVER.PHASES:
            # Compute result
            det_result = compute_result["perframe"](
                cfg,
                det_gt_targets,
                det_pred_scores,
            )
            log.append(
                "test det_loss: {:.5f} det_mAP: {:.5f}".format(
                    det_losses["test"] / len(data_loaders["test"].dataset),
                    det_result["mean_AP"],
                )
            )
        log.append(
            "running time: {:.2f} sec".format(
                end - start,
            )
        )
        if do_logging:
            logger.info(" | ".join(log))
            # Save checkpoint for model and optimizer
            checkpointer.save(epoch, models, optimizers)

        # Shuffle dataset for next epoch
        data_loaders["train"].dataset.shuffle(epoch)
        du.synchronize()
