## Overview
This repository contains the official open-source implementation of PhyDiffNet and RaPVFormer, the two-stage framework proposed in our paper
"High-fidelity full-sky video prediction for photovoltaic ramp event forecasting".

The project integrates physics-informed video prediction, generative diffusion modeling, and transformer-based ramp-aware PV forecasting for ultra-short-term solar forecasting.


## File Descriptions

| File Name | Function Overview |
|---|---|
| **`PhyDNet.py`** | Training and evaluation CLI for the paper-aligned PhyDNet stage. Builds the physics/residual recurrent model, applies scheduled teacher forcing and differential-moment regularization, and writes start-index-aligned coarse future-frame predictions. |
| **`RaPVFormer.py`** | Defines RaPVFormer, its PV regression and auxiliary ramp-classification losses, equal-weight real/generated dual-view training, and generated-view-only validation and testing. |
| **`constrain_moments.py`** | Implements the kernel-to-moment (`K2M`) transform used to constrain PhyCell's learned physical convolution kernels. |
| **`diff_modules.py`** | Provides diffusion-specific layers, linear self/cross-attention, conditional encoders, the Gaussian diffusion process, and the diffusion trainer. |
| **`generate_phydiffnet.py`** | Loads a trained diffusion checkpoint, refines PhyDNet coarse forecasts, and saves the generated 16-frame sequences with their `eval_stidx` and split metadata. |
| **`pipeline.py`** | Exposes the leakage-safe end-to-end `PVRampPipeline`, connecting PhyDNet, conditional diffusion, and RaPVFormer without accepting future observed RGB frames. |
| **`prepare_diffusion_data.py`** | Creates the train-only conditional-diffusion HDF5 dataset containing observed futures, coarse predictions, four historical frames, start indices, and split-manifest metadata. |
| **`rnn_models.py`** | Defines PhyCell, ConvLSTM, the dual-branch PhyDNet encoder/decoder, channel attention, SSIM, and the composite frame loss. |
| **`utilities.py`** | Centralizes dataset classes, synchronized spatial augmentation, day-level split manifests, normalization, device transfer, prediction HDF5 I/O, checkpoint metadata, logging, and reproducibility helpers. |
| **`video_conditional_diffusion.py`** | Defines the conditional 3D U-Net and the `paper`/`legacy` diffusion architectures, and provides the diffusion-training CLI. |
| **`requirements.txt`** | Lists the Python runtime and test dependencies, including PyTorch 2.6, torchvision, NumPy, pandas, h5py, einops, and pytest. |
| **`supplementary-file.pdf`** | Supplementary document containing detailed model hyperparameters, training/inference protocols, video-quality metric definitions, and dataset links. |
| **`LICENSE.txt`** | Repository license terms. |
| **`README.md`** | Project overview, complete file inventory, dataset resources, environment requirements, training workflow, and method-to-code correspondence notes. |

## Dataset Resources
Sky Image and Photovoltaic Power Generation Dataset (SKIPP'D):

- 2017: https://purl.stanford.edu/sm043zf7254
- 2018: https://purl.stanford.edu/fb002mq9407
- 2019: https://purl.stanford.edu/jj716hx9049

## Requirements

- Python 3.12
- PyTorch 2.6

Install the remaining dependencies from `requirements.txt`.



