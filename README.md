## Conditional Flow Matching for Spatio-Temporal Graphs

This code is a PyTorch Lightning implementation of the SIGSPATIAL'23 paper "DiffSTG: Probabilistic Spatio-Temporal Graph Forecasting with Denoising Diffusion Models". [arXiv](https://arxiv.org/abs/2301.13629)

Furthermore, this repository implements a conditional flow matching approach that replaces the standard diffusion used in DiffSTG.

## Model backbone

The backbone used in both approaches is UGNet and the overall worflow looks like:
![image](./img/model.png)

## Results on the PEMS Dataset:

| Model                    | MAE   | RMSE  |
| ------------------------ | ----- | ----- |
| CFMSTG                   | 16.85 | 25.75 |
| DiffSTG (DDPM 200 Steps) | 17.12 | 25.78 |
| DiffSTG (DDIM 20 Steps)  | 21.27 | 30.45 |

## Run

To run the training script, you need to install the required packages. We recommend using poetry:

```
# Install pipx and install poetry
python -m pip install pipx
pipx install poetry
pipx ensurepath

# Install packages
poetry install

# Enable the virtual environment
poetry shell

# Run the script
python train_lightning.py
```

Alternatively, we provide a `requirements.txt` file that you can use to install the packages.

