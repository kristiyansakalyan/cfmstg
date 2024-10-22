"""This module implements the DiffSTG model."""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from cfmstg.data.datamodule import TrafficDataModule
from cfmstg.model.submodels.ugnet import UGnet
from cfmstg.task.forecasting import ForecastingModel
from cfmstg.utils.common import EvaluationMode


class CFMSTG(ForecastingModel):

    def __init__(self: "CFMSTG", config: DictConfig) -> None:
        """
        Initializes the DiffSTG model with the given configuration.

        Parameters
        ----------
        config : easydict.EasyDict
            Configuration object with model hyperparameters.
        """
        super().__init__()
        self.config = config
        # Number of steps in the forward process
        self.N = config.N
        # Steps in the sample process
        self.sample_steps = config.sample_steps
        # Sampling strategy
        self.sample_strategy = self.config.sample_strategy
        # Minimum variance
        self.sigma_min = config.sigma_min
        self.device = torch.device(config.device_name)
        self.mask_ratio = config.mask_ratio

        # Initialize epsilon model (e.g., UGnet)
        if config.epsilon_theta == "UGnet":
            self.eps_model = UGnet(config).to(self.device)

    def forward(self: "CFMSTG", input: tuple, n_samples: int = 1) -> torch.Tensor:
        # TODO: Sample loop
        return self.evaluate(input, n_samples)

    def phi_t(
        self: "CFMSTG", x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return (1 - (1 - self.sigma_min) * t) * x_0 + (t * x_1)

    def u_t(self: "CFMSTG", x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return x_1 - ((1 - self.sigma_min) * x_0)

    def loss(self: "CFMSTG", x_1: torch.Tensor, c: tuple) -> torch.Tensor:
        # Sample time step
        # TODO: This won't work, must be reshaped s.t. it is broadcasted correctly
        t = torch.rand(1, x_1.shape[0], device=x_1.device)
        # Sample random noise as initial point
        x_0 = torch.randn_like(x_1)
        # Compute the sample x at time t
        x_t = self.phi_t(x_0, x_1, t)
        # Compute the vector field
        u_t = self.u_t(x_0, x_1)
        # Predict the vector field
        v_t = self.eps_model(x_t, t, c)
        # Measure the MSE Loss between the actual and predicted vector fields.
        return F.mse_loss(u_t, v_t)

    def eval_lightning(
        self: "CFMSTG",
        batch: tuple[torch.Tensor, torch.Tensor, int, int],
        datamodule: TrafficDataModule,
        sample_steps: int | None = None,
        sample_strategy: str | None = None,
        mode: EvaluationMode = "val",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Eval step
        if sample_steps is not None and sample_strategy is not None:
            self.set_ddim_sample_steps(sample_steps)
            self.set_sample_strategy(sample_strategy)

        # target:(B,T,V,1), history:(B,T,V,1), pos_w: (B,1), pos_d:(B,T,1)
        future, history, pos_w, pos_d = batch

        # in cpu (B, T, V, F), T =  T_h + T_p
        x = torch.cat((history, future), dim=1).to(future.device)
        # (B, T, V, F)
        x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(
            future.device
        )
        # (B, F, V, T)
        x = x.transpose(1, 3)
        # (B, F, V, T)
        x_masked = x_masked.transpose(1, 3)

        n_samples = 1 if mode == "val" else self.config.n_samples
        # (B, n_samples, F, V, T)
        x_hat = self((x_masked, pos_w, pos_d), n_samples)

        # No clue why we do that here.
        if x_hat.shape[-1] != (self.config.T_h + self.config.T_p):
            x_hat = x_hat.transpose(2, 4)

        x = datamodule.clean_dataset.reverse_normalization(x)
        x_hat = datamodule.clean_dataset.reverse_normalization(x_hat)

        f_x, f_x_hat = (
            x[:, :, :, -self.config.T_p :],
            x_hat[:, :, :, :, -self.config.T_p :],
        )  # future

        # y_true: (B, T_p, V, D)
        _y_true_ = f_x.transpose(1, 3)
        # y_pred: (B, n_samples, T_p, V, D)
        _y_pred_ = f_x_hat.transpose(2, 4)
        _y_pred_ = torch.clip(_y_pred_, 0, torch.inf)

        return _y_true_, _y_pred_

    def loss_lightning(
        self: "CFMSTG",
        batch: tuple[torch.Tensor, torch.Tensor, int, int],
    ) -> torch.Tensor:
        # TODO: This in theory should work out of the box
        # future: (B, T_p, V, F), history: (B, T_h, V, F)
        future, history, pos_w, pos_d = batch

        # get x_1 all
        x = torch.cat((history, future), dim=1).to(self.device)  #  (B, T, V, F)

        # get x_1 masked
        mask = torch.randint_like(history, low=0, high=100) < int(
            # mask the history in a ratio with mask_ratio
            self.mask_ratio
            * 100
        )
        history[mask] = 0
        x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(
            self.device
        )  # (B, T, V, F)

        # reshape
        x = x.transpose(1, 3)  # (B, F, V, T)
        x_masked = x_masked.transpose(1, 3)  # (B, F, V, T)

        # loss calculate
        return 10 * self.loss(x, (x_masked, pos_w, pos_d))
