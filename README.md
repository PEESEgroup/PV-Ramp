## Overview
This repository contains the official open-source implementation of PhyDiffNet and RaPVFormer, the two-stage framework proposed in our paper
“High-fidelity full-sky video prediction for photovoltaic ramp event forecasting”.

The project integrates physics-informed video prediction, generative diffusion modeling, and transformer-based ramp-aware PV forecasting to deliver state-of-the-art ultra–short-term solar forecasting.


## File Descriptions

| File Name                     | Function Overview |
|------------------------------|-------------------|
| **PhyDNet.py**               | Implements the physics-informed video prediction module (PhyDNet), including the dual-branch architecture (physics-guided PDE branch + residual ConvLSTM), decoders, training flow, and SSIM-based losses. Produces coarse future sky frames. |
| **video_conditional_diffusion.py** | Implements the conditional diffusion model used to refine PhyDNet’s coarse predictions. Uses DDPM forward/reverse processes with a U-Net noise predictor conditioned on historical frames. Produces high-fidelity sky video frames. |
| **RaPVFormer.py**           | Implements the transformer-based PV forecasting model (RaPVFormer). Encodes historical/predicted frames and PV output, uses self-attention + cross-attention, and outputs multi-step PV predictions & ramp classifications. |
| **rnn_models.py**           | Contains reusable RNN/ConvLSTM modules, PDE-guided cells, sequence encoders/decoders, and auxiliary temporal modeling components used by PhyDNet and other modules. |
| **constrain_moments.py**    | Implements moment-based constraints and physical regularization losses used in PhyDNet. |
| **utilities.py**            | Provides utility functions such as dataset, sun-mask generation, logging configuration, and other general helpers used across the framework. |

## Dataset Resources
Sky Image and Photovoltaic Power Generation Dataset (SKIPP’D): 
- 2017: https://purl.stanford.edu/sm043zf7254
- 2018: https://purl.stanford.edu/fb002mq9407
- 2019: https://purl.stanford.edu/jj716hx9049

## Requirements

- Python 3.12
- PyTorch 2.6

