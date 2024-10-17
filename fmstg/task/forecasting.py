from abc import ABC, abstractmethod
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from fmstg.data.datamodule import TrafficDataModule


class ForecastingModel(ABC, nn.Module):
    """
    Abstract base class for forecasting models.

    This class forces subclasses to implement a 'loss_lightning' method, which calculates the
    loss in a PyTorch Lightning context.
    """

    @abstractmethod
    def loss_lightning(self, batch: tuple) -> torch.Tensor:
        """
        Abstract method to compute the loss in a PyTorch Lightning training loop.

        Parameters
        ----------
        batch : tuple
            The input batch containing the data for training.

        Returns
        -------
        torch.Tensor
            The computed loss.

        Raises
        ------
        NotImplementedError
            If this method is not implemented by the subclass.
        """
        pass

    @abstractmethod
    def eval_lightning(
        self,
        batch: tuple,
        datamodule: pl.LightningDataModule,
        sample_steps: int | None = None,
        sample_strategy: str | None = None,
    ) -> torch.Tensor:
        """
        Abstract method to compute the loss in a PyTorch Lightning training loop.

        Parameters
        ----------
        batch : tuple
            The input batch containing the data for training.
        datamodule: pl.LightningDataModule
            The datamodule that can be used to reverse the normalization for evaluation.
        sample_steps : int | None
            The number of sample steps to be used.
        sample_strategy : str | None
            The sample strategy to be used.

        Returns
        -------
        torch.Tensor
            The computed loss.

        Raises
        ------
        NotImplementedError
            If this method is not implemented by the subclass.
        """
        pass


def get_lr_scheduler(
    config: dict, optimizer: torch.optim.Optimizer
) -> dict[str, Any] | None:
    """Get the learning rate scheduler.

    Parameters
    ----------
    config : dict
        Training config.
    optimizer : torch.optim.Optimizer
        The optimizer to which the lr_scheduler has to be attached.

    Returns
    -------
    dict[str, Any] | None
        Learning scheduler configuration for PyTorch Lightning

    Raises
    ------
    ValueError
        If the learning rate scheduler is not supported.
    """
    lr_scheduler_name = config["training_hparams"]["lr_scheduler"]["name"]

    if lr_scheduler_name == "ReduceLROnPlateau":
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.training_hparams.lr_scheduler.factor,
            patience=config.training_hparams.lr_scheduler.patience,
        )
        return {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": config.training_hparams.lr_scheduler.monitor,
        }
    if lr_scheduler_name == "StepLR":
        lr_scheduler = StepLR(
            optimizer,
            step_size=config.training_hparams.lr_scheduler.step_size,
            gamma=config.training_hparams.lr_scheduler.gamma,
        )
        return {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
    else:
        raise ValueError(
            f"The provided learning rate scheduler: '{lr_scheduler_name}' is not supported."
        )


class ForecastingTask(pl.LightningModule):
    def __init__(
        self: "ForecastingTask",
        model: ForecastingModel,
        datamodule: TrafficDataModule,
        config: DictConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.config = config
        # TODO: RMSE? CRPS?
        # TODO: How can I also measure the time for train, val, test!? TQDM Progress bar?
        metrics = tm.MetricCollection(
            {
                "mae": tm.MeanAbsoluteError(),
                "mse": tm.MeanSquaredError(),
            }
        )
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # https://github.com/Lightning-AI/lightning/pull/16520.
        # The new on_validation_epoch_end hook makes us save the outputs
        # in an attribute instead of having them passed as an argument.
        self.val_output_list = []

    def training_step(
        self: "ForecastingTask",
        batch: tuple[torch.Tensor, torch.Tensor, int, int],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        loss = self.model.loss_lightning(batch)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss}

    def validation_step(
        self: "ForecastingTask",
        batch: tuple[torch.Tensor, torch.Tensor, int, int],
        batch_idx: int,
    ) -> None:
        _y_true_, _y_pred_ = self.model.eval_lightning(
            batch,
            self.datamodule,
            self.config.model.eval.sample_steps,
            self.config.model.eval.sample_strategy,
        )
        metrics = self.val_metrics(
            _y_true_.contiguous(), _y_pred_.squeeze(1).contiguous()
        )
        # https://github.com/Lightning-AI/lightning/pull/16520.
        # The new on_validation_epoch_end hook makes us save the outputs
        # in an attribute instead of having them passed as an argument.
        # Add output
        self.val_output_list.append(metrics["val/mse"])

        # Log validation metrics for the current batch
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)

        # Return validation MSE for logging and later use by the scheduler
        return {"val/mse": metrics["val/mse"]}

    def on_before_optimizer_step(
        self: "ForecastingTask", optimizer: torch.optim.Optimizer
    ) -> None:
        # Compute and log gradient norms (L2 norm)
        grad_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(p.grad.detach(), 2)
                    for p in self.model.parameters()
                    if p.grad is not None
                ]
            )
        )
        self.log(
            "train/grad_norm",
            grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

    def on_validation_epoch_end(self: "ForecastingTask") -> None:
        # Gather all the validation steps to get the average validation MSE
        avg_val_mse = torch.stack(self.val_output_list).mean()

        # https://github.com/Lightning-AI/lightning/pull/16520.
        # The new on_validation_epoch_end hook makes us save the outputs
        # in an attribute instead of having them passed as an argument.
        # Reset list
        self.val_output_list = []

        # Log the averaged validation MSE, which will be used by the scheduler
        self.log("val/mse", avg_val_mse, on_epoch=True, prog_bar=True, logger=True)

        # Also log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/learning_rate", lr, on_epoch=True, prog_bar=True, logger=True)

    def test_step(
        self: "ForecastingTask",
        batch: tuple[torch.Tensor, torch.Tensor, int, int],
        batch_idx: int,
    ) -> None:
        _y_true_, _y_pred_ = self.model.eval_lightning(batch, self.datamodule)
        self.log_dict(
            self.test_metrics(_y_true_, _y_pred_),
            prog_bar=True,
            on_step=True,
            logger=True,
        )

    def configure_optimizers(self: "ForecastingTask") -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training_hparams.learning_rate,
            weight_decay=self.config.training_hparams.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": get_lr_scheduler(self.config, optimizer),
        }
