# Rekognition Stochastic Backpropagation - CVPR 2022
This repo is the official implementation of ["Stochastic Backpropagation: A Memory Efficient Strategy for Training Video Models"](https://arxiv.org/abs/2203.16755).
It was accepted to CVPR 2022 (Oral).

Stochastic Backpropagation (SBP) is a memory saving technique for training video models. It can save up to 80% the memory with minor performance drop. It is very efficient for tasks that take hundreds of frames as input such as temporal/online action detection.

## Installation

### Requirements

-   linux. python \>= 3.6.
-   pytorch == 1.9.1
-   other requirements in `requirements.txt`. Install using `pip install -r requirements.txt `.
If you occur the error that some package can not be found, you can simply install the missing pacakges using `pip`.

### Add to `PYTHONPATH`:

Add the `src` directory to `PYTHONPATH`.
```
export PYTHONPATH=`pwd`/src:$PYTHONPATH
```

## Action Recognition

We validate SBP with video Swin model. The SBP implementations are inside `src/models/archs/vswin/sbp_ops.py`.

### Data preparation.
1. Download and *unzip* K400.
    - [videos(134GB)](https://yzaws-data-log.s3.amazonaws.com/data/Kinetics/kinetics400.zip). Unzip using `unzip kinetics400.zip`.
    - Annotations: [train](https://yzaws-data-log.s3.amazonaws.com/data/Kinetics/k400_train.txt), [val](https://yzaws-data-log.s3.amazonaws.com/data/Kinetics/k400_val.txt)
2. link the k400 directory to `data/kinetics400` using `ln -s {k400_path} data/kinetics400`. Replace `{k400_path}` with the actual directory that stores your K400 data.

### Imagenet pretrained weights
Download Imagenet pretrained SWIN weights from [official repo](https://github.com/microsoft/Swin-Transformer) and put into `pretrained_models/swin`.
```sh
mkdir -p pretrained_models/swin
cd pretrained_models/swin
# swin_t weights
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
# swin_b weights
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
cd ../..
```

### Train and test

In the following example, we train video swin-tiny with SBP of `keep_ratio=0.5` (== 1 / gd_downsample). Currently only support `gd_downsample` with `2^n`, i.e. 2, 4, 8.
The checkpoint dir will be `./checkpoints/actrec-k400/swin_t-sbp_0.5`. This setting can be trained on a machine with 4x 8GB GPUs. If you change the batch size, please linearly scale the learning rate (`SOLVER.OPTIMIZER.LR`).

```
# train.
python tools/train_net.py --config_file configs/kinetics/k400-swin_tiny.yaml \
    MEMSAVE.ENABLE True MEMSAVE.GRADDROP_CFG.gd_downsample 2 \
    SESSION swin_t-sbp_0.5

# test with 12 views.
python tools/test_net.py --config_file configs/kinetics/k400-swin_tiny.yaml \
    MEMSAVE.ENABLE True MEMSAVE.GRADDROP_CFG.gd_downsample 2 \
    SESSION swin_t-sbp_0.5 \
    MODEL.CHECKPOINT checkpoints/actrec_k400/swin_t-sbp_0.5/ckpts/epoch-30.pth
```

You can train video swin-base with config `configs/kinetics/k400-swin_base.yaml`.

## Online Action Detection

We validate SBP with [LSTR](https://github.com/amazon-research/long-short-term-transformer) for online action detection.
The key implementation of SBP for LSTR is in `src/lstr/trainer/partial_feedback.py` of function `partial_feedback_forward`.

### Data preparation.
Please refer to [LSTR](https://github.com/amazon-research/long-short-term-transformer).
- For each video, extract rgb frames with 4fps and save to `{video_name}.npy` file in directory `data/THUMOS/rgb_npy`. The saved format is `THWC`.
- Generate the `data/THUMOS/target_perframe` using the provide scripts in `tools/lstr/perframe_label_generation.py`.

### Download pretrained weights
Download pretrained weights from [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsn/README.md) and put into `pretrained_models/mmaction2_resnet`.

```sh
mkdir -p pretrained_models/mmaction2_resnet
cd pretrained_models/mmaction2_resnet
wget https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth
cd ../..
```

### Train and test

```
# train.
python tools/lstr/train_net.py --config_file configs/thumos/sbp_e2e_lstr_128_32.yaml \
    SESSION sbp

# evaluation.
python tools/lstr/test_net.py --config_file configs/thumos/sbp_e2e_lstr_128_32.yaml \
    SESSION sbp \
    MODEL.CHECKPOINT checkpoints/thumos-lstr/sbp/ckpts/epoch-25.pth
```

## Credits
Our implementation is inspired by several open-sourced work, including:
 - [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
 - [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)
 - [LSTR](https://github.com/amazon-research/long-short-term-transformer)

Thanks for their great work!

## Citation

If you find this project useful for your research, please use the
following BibTeX entry.

    @article{cheng2022stochastic,
      title={Stochastic Backpropagation: A Memory Efficient Strategy for Training Video Models},
      author={Cheng, Feng and Xu, Mingze and Xiong, Yuanjun and Chen, Hao and Li, Xinyu and Li, Wei and Xia, Wei},
      journal={arXiv preprint arXiv:2203.16755},
      year={2022}
    }
