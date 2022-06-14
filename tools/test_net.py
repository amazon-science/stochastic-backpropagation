import os
import sys
sys.path.insert(0, 'src')
from time import time
from tqdm import tqdm
from easydict import EasyDict

import torch
import numpy as np
from torchmetrics import Accuracy, MetricCollection

from dlhammer.bootstrap import bootstrap, logger
from dlhammer.nested import nested_call
from dlhammer.env import setup_random_seed
from dlhammer.checkpointer import setup_checkpointer

from factory import build_data_loader, build_model


phase = 'test'


def main():
    cfg = bootstrap(print_cfg=False)

    # Setup ddp
    cfg.DDP.ENABLE = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup seed
    setup_random_seed(cfg.SEED)

    # Setup checkpointer
    checkpointer = setup_checkpointer(cfg, phase=phase)

    # Build model
    model = build_model(cfg, device)
    nested_call(model, 'eval')

    # Load pretrained model
    checkpointer.load_models(model)

    # build metrics
    metrics = EasyDict()
    metrics_obj = MetricCollection({
        'top1': Accuracy(top_k=1),
        'top5': Accuracy(top_k=5),
    })
    metrics[phase] = metrics_obj.clone(prefix=f'{phase}_')

    # Testing with 12 view
    logger.info(f'===== Testing with 12 views on epoch {cfg.SOLVER.START_EPOCH} =====')
    crops = ['top_left', 'center', 'bottom_right']
    cfg.DATA.TEST.NUM_CLIPS = 4
    cfg.DATA.TEST.CLIP_SAMPLE_STRATEGY = 'uniform'

    n_crops = len(crops)
    total_preds = [[] for _ in range(n_crops)]
    total_gts = [[] for _ in range(n_crops)]

    # Run test
    start = time()
    for i, crop in enumerate(crops):
        cfg.DATA.TEST.CROP_POS = crop
        data_loader = build_data_loader(cfg, phase)

        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f'Testing epoch {cfg.SOLVER.START_EPOCH}')
            for inputs, labels in pbar:
                inputs = [x.to(device, non_blocking=True) for x in inputs]
                labels = labels.to(device, non_blocking=True)

                # Step
                logits = model(inputs)

                logits = logits.softmax(-1).cpu().tolist()  # shape: (B, C)
                labels = labels.cpu().tolist() # shape: (B, 1)

                total_preds[i].extend(logits)
                total_gts[i].extend(labels)
    end = time()

    # Aggregate results
    total_preds = np.array(total_preds)  # shape: (n_crop, B, C) where B = n_samples x num_clips
    total_preds = np.reshape(
        total_preds,
        (n_crops, -1, cfg.DATA.TEST.NUM_CLIPS, cfg.MODEL.SWIN.HEAD.NUM_CLASSES),
    )
    total_preds = np.transpose(
        total_preds, [0, 2, 1, 3]
    )  # shape: (n_crop, num_clips, num_samples, num_classes)
    total_preds = np.mean(total_preds, (0, 1))  # shape: (num_samples, num_classes)

    total_gts = np.array(total_gts)  # shape (n_crop, B)
    total_gts = np.reshape(
        total_gts,
        (n_crops, -1, cfg.DATA.TEST.NUM_CLIPS),
    )
    total_gts = np.transpose(
        total_gts, [0, 2, 1]
    )  # shape: (n_crop, num_clips, num_samples)

    assert np.array_equal(
        total_gts[0, 0, :], total_gts[n_crops - 1, cfg.DATA.TEST.NUM_CLIPS - 1, :]
    ), 'Found errors in results aggregation'
    total_gts = total_gts[0, 0, :]  # shape: (num_samples)

    # To tensor
    total_preds = torch.tensor(total_preds)
    total_gts = torch.tensor(total_gts)
    metrics[phase](total_preds, total_gts)
    total_metrics = metrics[phase].compute()
    total_metrics = {
        key: float(value)
        for key, value in total_metrics.items()
    }

    # Output log for current epoch
    log = []
    log.append(f'Epoch {cfg.SOLVER.START_EPOCH:2}')
    for key, value in total_metrics.items():
        log.append(f'{key}: {value:.5f}')
    log.append(f'running time: {end - start:.3f} sec')
    logger.info(' | '.join(log))


if __name__ == "__main__":
    main()
