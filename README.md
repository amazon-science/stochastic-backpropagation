# Rekognition Stochastic Backpropagation - NeurIPS 2022

Official PyTorch implementation for our paper at NeurIPS 2022:

**An In-depth Study of Stochastic Backpropagation** [[arXiv]](https://arxiv.org/abs/2210.00129)

By Jun Fang, Mingze Xu, Hao Chen, Bing Shuai, Zhuowen Tu, Joseph Tighe

Stochastic Backpropagation (SBP) is a memory-efficient technique for training computer vision models. 
It can save up to 40% of GPU memory with less than 1% accuracy degradation for training image recognition models.


## Installation

The results in the paper are produced with `Python==3.7.10 torch==1.8.1+cu111 timm==0.3.2`.

```
pip install torch==1.8.0+cu111 
pip install timm==0.3.2 tensorboardX six
```

## Training
See `run-train.sh` for training instructions.

## Results on ImageNet

|    Network    | Keep-ratio | Batch size | Memory (MB / GPU) | Top-1 accuracy (%) |
|:-------------:|:----------:|:----------:|:-----------------:|:------------------:|
|    ViT-Tiny   |   no SBP   |     256    |        8248       |        73.68       |
|    ViT-Tiny   |     0.5    |     256    |    5587 (0.68×)   |    73.09 (-0.59)   |
|    ViT-Base   |   no SBP   |     64     |       10083       |        81.22       |
|    ViT-Base   |     0.5    |     64     |    7436 (0.74×)   |    80.62 (-0.60)   |
| ConvNeXt-Tiny |   no SBP   |     128    |       12134       |        82.1        |
| ConvNeXt-Tiny |     0.5    |     128    |    7059 (0.58×)   |    81.61 (-0.49)   |
| ConvNeXt-Base |   no SBP   |     64     |       14130       |        83.8        |
| ConvNeXt-Base |     0.5    |     64     |    8758 (0.62×)   |    83.27 (-0.53)   |



## Acknowledgement
This repository is built using the [timm 0.3.2](https://github.com/rwightman/pytorch-image-models) library and the [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) repository. Thanks for their great work!


## Citation

If you find this project useful for your research, please cite our work by using the following BibTeX entry.

    @article{fang2022depth,
      title={An In-depth Study of Stochastic Backpropagation},
      author={Fang, Jun and Xu, Mingze and Chen, Hao and Shuai, Bing and Tu, Zhuowen and Tighe, Joseph},
      journal={arXiv preprint arXiv:2210.00129},
      year={2022}
    }
    
    @article{cheng2022stochastic,
      title={Stochastic Backpropagation: A Memory Efficient Strategy for Training Video Models},
      author={Cheng, Feng and Xu, Mingze and Xiong, Yuanjun and Chen, Hao and Li, Xinyu and Li, Wei and Xia, Wei},
      journal={arXiv preprint arXiv:2203.16755},
      year={2022}
    }