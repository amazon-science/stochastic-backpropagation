import os.path as osp
import pickle as pkl

import numpy as np
import torch
from easydict import EasyDict
from torch import nn
from tqdm import tqdm

from ...datasets import build_dataset
from ...evaluation import compute_result
from ..partial_feedback import vanilla_forward


def do_e2e_perframe_det_batch_inference(cfg, model, device, logger):
    # Setup model to test mode
    # Setup model on multiple GPUs
    if torch.cuda.device_count() > 1:
        model = EasyDict({k: nn.DataParallel(v) for k, v in model.items()})

    [m.eval() for _, m in model.items()]

    data_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg, phase="test", tag="BatchInference"),
        batch_size=cfg.DATA_LOADER.TEST_BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )

    # Collect scores and targets
    pred_scores = {}
    gt_targets = {}

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="BatchInference")
        for batch_idx, data in enumerate(pbar, start=1):
            target = data[-4]

            if cfg.INPUT.MODALITY == "visual":
                data[1] = None
            elif cfg.INPUT.MODALITY == "motion":
                data[0] = None

            inp = [x.to(device) if x is not None else x for x in data[:3]]
            logits = vanilla_forward(model, inp)
            score = logits.softmax(dim=-1).cpu().numpy()

            for bs, (session, query_indices, num_frames) in enumerate(zip(*data[-3:])):
                if session not in pred_scores:
                    pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                if session not in gt_targets:
                    gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))

                if query_indices[0] == 0:
                    pred_scores[session][query_indices] = score[bs]
                    gt_targets[session][query_indices] = target[bs]
                else:
                    pred_scores[session][query_indices[-1]] = score[bs][-1]
                    gt_targets[session][query_indices[-1]] = target[bs][-1]

    # Save scores and targets
    pkl.dump(
        {
            "cfg": cfg,
            "perframe_pred_scores": pred_scores,
            "perframe_gt_targets": gt_targets,
        },
        open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + ".pkl", "wb"),
    )

    # Compute results
    result = compute_result["perframe"](
        cfg,
        np.concatenate(list(gt_targets.values()), axis=0),
        np.concatenate(list(pred_scores.values()), axis=0),
    )
    logger.info(
        "Action detection perframe m{}: {:.5f}".format(
            cfg.DATA.METRICS, result["mean_AP"]
        )
    )
