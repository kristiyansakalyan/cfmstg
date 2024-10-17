"""This module implements the DiffSTG model."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from fmstg.data.datamodule import TrafficDataModule
from fmstg.model.submodels.ugnet import UGnet
from fmstg.task.forecasting import ForecastingModel
from fmstg.utils.common import gather


class DiffSTG(ForecastingModel):
    """
    Diffusion Spatio-Temporal Graph Model (DiffSTG).

    This model uses a diffusion process to generate predictions for spatio-temporal tasks. The forward process
    gradually adds noise to the input, while the reverse process denoises the input based on conditions.

    Attributes
    ----------
    config : DictConfig
        The configuration object containing model hyperparameters.
    N : int
        Number of steps in the forward diffusion process.
    sample_steps : int
        Number of steps during the sampling process.
    sample_strategy : str
        The sampling strategy used for generating samples (e.g., 'ddim', 'ddpm').
    device : torch.device
        Device on which the model is running.
    beta_start : float
        Starting value of the noise schedule.
    beta_end : float
        Ending value of the noise schedule.
    beta_schedule : str
        Type of noise schedule ('uniform' or 'quad').
    eps_model : nn.Module
        The epsilon model (e.g., UGnet) that predicts the noise at each step.
    beta : torch.Tensor
        The noise schedule values.
    alpha : torch.Tensor
        The corresponding alpha values for the noise schedule.
    alpha_bar : torch.Tensor
        Cumulative product of alpha over time.
    sigma2 : torch.Tensor
        Variance values for the diffusion process.
    """

    def __init__(self: "DiffSTG", config: DictConfig) -> None:
        """
        Initializes the DiffSTG model with the given configuration.

        Parameters
        ----------
        config : easydict.EasyDict
            Configuration object with model hyperparameters.
        """
        super().__init__()
        self.config = config
        self.N = config.N  # Number of steps in the forward process
        self.sample_steps = config.sample_steps  # Steps in the sample process
        self.sample_strategy = self.config.sample_strategy  # Sampling strategy
        self.device = torch.device(config.device_name)
        self.beta_start = config.get("beta_start", 0.0001)
        self.beta_end = config.get("beta_end", 0.02)
        self.beta_schedule = config.beta_schedule
        self.mask_ratio = config.mask_ratio

        # Initialize epsilon model (e.g., UGnet)
        if config.epsilon_theta == "UGnet":
            self.eps_model = UGnet(config).to(self.device)

        # Create $$ \beta_1, \dots, \beta_T $$
        if self.beta_schedule == "uniform":
            self.beta = torch.linspace(self.beta_start, self.beta_end, self.N).to(
                self.device
            )
        elif self.beta_schedule == "quad":
            self.beta = (
                torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.N) ** 2
            )
            self.beta = self.beta.to(self.device)
        else:
            raise NotImplementedError

        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta

    def q_xt_x0(
        self: "DiffSTG",
        x0: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Samples from the forward process
        $$ q(x_t | x_0) \sim N(x_t; \sqrt{\bar{\alpha_t}} * x_0, (1 - \bar{\alpha_t}) * I_D) $$

        Parameters
        ----------
        x0 : torch.Tensor
            The initial input tensor (batch, features, nodes, time).
        t : torch.Tensor
            The current timestep.
        eps : torch.Tensor | None
            The noise to be added (if not provided, random noise will be used), by default None.

        Returns
        -------
        torch.Tensor
            The sampled tensor $x_t$ from the forward process.
        """
        if eps is None:
            eps = torch.randn_like(x0)

        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)

        return mean + eps * (var**0.5)

    def p_sample(
        self: "DiffSTG", xt: torch.Tensor, t: torch.Tensor, c: tuple
    ) -> torch.Tensor:
        """
        Samples from the reverse process
        - $$ p(x_{t-1} | x_t, c) = \mathcal{N}(x_{t-1}; \mu_t, \sigma_t^2 I) $$
        Where:
        - $$ \mu_t = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, c) \right) $$
        - $$ \sigma_t^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t $$
        Finally, to sample from \( p(x_{t-1} | x_t, c) \):
        - $$ x_{t-1} = \mu_t + \sqrt{\sigma_t^2} \cdot \epsilon $$
        Where $$ ( \epsilon \sim \mathcal{N}(0, I) ) $$
        is standard Gaussian noise.
        The complete equation for the reverse sampling process:
        - $$ p(x_{t-1} | x_t, c) = \mathcal{N}\left(x_{t-1}; \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, c) \right), \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \cdot I \right) $$



        Parameters
        ----------
        xt : torch.Tensor
            The current noisy input tensor $x_t$.
        t : torch.Tensor
            The current timestep.
        c : tuple
            The condition information used for sampling (e.g., masked input).

        Returns
        -------
        torch.Tensor
            The sampled tensor $x_{t-1}$ from the reverse process.
        """
        eps_theta = self.eps_model(xt, t, c)  # c is the condition
        alpha_coef = 1.0 / (gather(self.alpha, t) ** 0.5)
        eps_coef = gather(self.beta, t) / (1 - gather(self.alpha_bar, t)) ** 0.5
        mean = alpha_coef * (xt - eps_coef * eps_theta)

        var = (
            (1 - gather(self.alpha_bar, t - 1))
            / (1 - gather(self.alpha_bar, t))
            * gather(self.beta, t)
        )
        eps = torch.randn(xt.shape, device=xt.device)

        return mean + eps * (var**0.5)

    def p_sample_loop(self: "DiffSTG", c: tuple) -> torch.Tensor:
        """
        Performs the reverse diffusion process iteratively for T steps.

        Parameters
        ----------
        c : tuple
            The masked input tensor (B, T, V, D), where T = T_h + T_p.

        Returns
        -------
        torch.Tensor
            The predicted output tensor (B, T, V, D).
        """
        x_masked, _, _ = c
        B, _, V, T = x_masked.shape
        with torch.no_grad():
            x = torch.randn(
                [B, self.config.F, V, T], device=self.device
            )  # Generate input noise
            for t in range(self.N, 0, -1):
                t = t - 1  # Index adjustment
                if t > 0:
                    x = self.p_sample(x, x.new_full((B,), t, dtype=torch.long), c)
        return x

    def p_sample_loop_ddim(
        self: "DiffSTG", c: tuple
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Performs the DDIM sampling for accelerated sampling.

        Parameters
        ----------
        c : tuple
            The masked input tensor (B, T, V, D).

        Returns
        -------
        tuple[list[torch.Tensor], list[torch.Tensor]]
            A tuple containing the intermediate samples and the predicted outputs.
        """
        x_masked, _, _ = c
        B, F, V, T = x_masked.shape
        N = self.N
        timesteps = self.sample_steps
        skip_type = self.beta_schedule

        if skip_type == "uniform":
            skip = N // timesteps
            seq = range(0, N, skip)
        elif skip_type == "quad":
            seq = np.linspace(0, np.sqrt(N * 0.8), timesteps) ** 2
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        x = torch.randn(
            [B, self.config.F, V, T], device=self.device
        )  # Generate input noise
        xs, x0_preds = generalized_steps(x, seq, self.eps_model, self.beta, c, eta=1)
        return xs, x0_preds

    def evaluate(self: "DiffSTG", input: tuple, n_samples: int = 2) -> torch.Tensor:
        """
        Generates samples using the selected sampling strategy.

        Parameters
        ----------
        input : tuple
            The input tensor (B, T, V, D).
        n_samples : int, optional
            The number of samples to generate, by default 2.

        Returns
        -------
        torch.Tensor
            The generated samples.
        """
        x_masked, _, _ = input
        B, F, V, T = x_masked.shape

        if self.sample_strategy == "ddim_multi":
            x_masked = (
                x_masked.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)
            )
            xs, _ = self.p_sample_loop_ddim((x_masked, _, _))
            x = xs[-1].reshape(B, n_samples, F, V, T)
            return x  # (B, n_samples, F, V, T)
        elif self.sample_strategy == "ddim_one":
            xs, _ = self.p_sample_loop_ddim((x_masked, _, _))
            x = xs[-n_samples:]
            return torch.stack(x, dim=1)
        elif self.sample_strategy == "ddpm":
            x_masked = (
                x_masked.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)
            )
            x = self.p_sample_loop((x_masked, _, _))
            x = x.reshape(B, n_samples, F, V, T)
            return x  # (B, n_samples, F, V, T)
        else:
            raise NotImplementedError

    def forward(self: "DiffSTG", input: tuple, n_samples: int = 1) -> torch.Tensor:
        """
        Forward method that generates samples from the model.

        Parameters
        ----------
        input : tuple
            The input tensor (B, T, V, D).
        n_samples : int, optional
            The number of samples to generate, by default 1.

        Returns
        -------
        torch.Tensor
            The generated samples.
        """
        return self.evaluate(input, n_samples)

    def loss(self: "DiffSTG", x0: torch.Tensor, c: tuple) -> torch.Tensor:
        """
        Computes the loss between the predicted noise and the true noise.

        Parameters
        ----------
        x0 : torch.Tensor
            The original input tensor (B, ...).
        c : tuple
            The condition, a tuple of torch tensors (feature, pos_w, pos_d).

        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        t = torch.randint(0, self.N, (x0.shape[0],), device=x0.device, dtype=torch.long)
        eps = torch.randn_like(x0)

        xt = self.q_xt_x0(x0, t, eps)
        eps_theta = self.eps_model(xt, t, c)
        return F.mse_loss(eps, eps_theta)

    def set_sample_strategy(self: "DiffSTG", sample_strategy: str) -> None:
        self.sample_strategy = sample_strategy

    def set_ddim_sample_steps(self: "DiffSTG", sample_steps: int) -> None:
        self.sample_steps = sample_steps

    def eval_lightning(
        self: "DiffSTG",
        batch: tuple[torch.Tensor, torch.Tensor, int, int],
        datamodule: TrafficDataModule,
        sample_steps: int | None = None,
        sample_strategy: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        n_samples = 1
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
        self: "DiffSTG",
        batch: tuple[torch.Tensor, torch.Tensor, int, int],
    ) -> torch.Tensor:
        future, history, pos_w, pos_d = (
            batch  # future:(B, T_p, V, F), history: (B, T_h, V, F)
        )

        # get x0
        x = torch.cat((history, future), dim=1).to(self.device)  #  (B, T, V, F)

        # get x0_masked
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

    def model_file_name(self) -> str:
        """
        Returns a filename based on the configuration parameters.

        Returns
        -------
        str
            The generated filename.
        """
        file_name = "+".join(
            [f"{k}-{self.config[k]}" for k in ["N", "T_h", "T_p", "epsilon_theta"]]
        )
        return f"{file_name}.dm4stg"


def compute_alpha(beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Computes the cumulative product of (1 - beta) for the given timesteps t.

    Parameters
    ----------
    beta : torch.Tensor
        The beta values (noise schedule).
    t : torch.Tensor
        The timesteps.

    Returns
    -------
    torch.Tensor
        The cumulative product of (1 - beta) for the given timesteps.
    """
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)


def generalized_steps(
    x: torch.Tensor,
    seq: list[int],
    model: nn.Module,
    b: torch.Tensor,
    c: tuple,
    **kwargs,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Implements the generalized DDIM sampling procedure.

    Parameters
    ----------
    x : torch.Tensor
        The initial noise tensor (B, F, V, T).
    seq : list[int]
        The sequence of timesteps for DDIM.
    model : nn.Module
        The epsilon model that predicts noise at each step.
    b : torch.Tensor
        The beta schedule.
    c : tuple
        Condition information for sampling.
    kwargs : dict
        Additional parameters (e.g., eta for controlling noise).

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor]]
        The intermediate samples and predicted outputs from the model.
    """
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device).long()
            next_t = (torch.ones(n) * j).to(x.device).long()
            at = compute_alpha(b, t)
            at_next = compute_alpha(b, next_t)
            xt = xs[-1].to(x.device)
            et = model(xt, t, c)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t)
            c1 = (
                kwargs.get("eta", 0)
                * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1**2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)

    return xs, x0_preds
