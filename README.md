# Mix-DDPM-Pytorch

**Mix-DDPM: Enhancing Diffusion Models through Fitting Mixture Noise with Global Stochastic Offset** <br>
*Hanzhang Wang, Deming Zhai, Xiong Zhou, Junjun Jiang, Xianming Liu.* <br>
ACM Transactions on Multimedia Computing, Communications, and Applications (ToMM), 2024.

## Reference

Citation:

```latex
@article{wang2024mix,
  title={Mix-DDPM: Enhancing Diffusion Models through Fitting Mixture Noise with Global Stochastic Offset},
  author={Wang, Hanzhang and Zhai, Deming and Zhou, Xiong and Jiang, Junjun and Liu, Xianming},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  publisher={ACM New York, NY}
}
```

## Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on.

## Preparing Data

The training code reads images from a directory of image files. In the [datasets](datasets) folder, we have provided instructions/scripts for preparing these directories for ImageNet, LSUN bedrooms, and CIFAR-10.

## Training

run bash file in train_sh

## Inference

run bash file in test_sh