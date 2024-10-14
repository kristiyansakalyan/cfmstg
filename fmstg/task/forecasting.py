from abc import ABC, abstractmethod

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics as tm


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

class ForecastingTask(pl.LightningModule):
    def __init__(self, model: ForecastingModel):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.model = model
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

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, int, int], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        loss = self.model.loss_lightning(batch)
        self.log("train/loss", loss, batch_size=batch[0].shape[0])
        return {"loss": loss}

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, int, int], batch_idx: int
    ) -> None:
        # TODO
        x, labels = batch
        self.log_dict(
            self.val_metrics(self.classify(x), labels),
            prog_bar=True,
            batch_size=x.shape[0],
        )

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, int, int], batch_idx: int
    ) -> None:
        # TODO
        x, labels = batch
        self.log_dict(
            self.test_metrics(self.classify(x), labels),
            prog_bar=True,
            batch_size=x.shape[0],
        )

    def configure_optimizers(self) -> torch.optim.optimizer.Optimizer:
        # TODO
        return torch.optim.Adam(self.model.parameters())
