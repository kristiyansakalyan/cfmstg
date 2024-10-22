import torch
from torch import Tensor
from torchmetrics import Metric

class RootMeanSquaredError(Metric):
    """
    Computes the Root Mean Squared Error (RMSE) between predictions and targets.

    RMSE is defined as the square root of the mean of squared differences between
    predicted and true values. It is widely used in regression tasks to measure 
    the average magnitude of the error between predicted and actual values.
    """

    def __init__(self: "RootMeanSquaredError", **kwargs) -> None:
        """
        Initializes the RootMeanSquaredError metric.

        This will set up two states for tracking:
        - `sum_squared_error`: the sum of squared differences between predictions and targets.
        - `total`: the total number of samples.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments, passed to the parent `Metric` class.
        """
        super().__init__(**kwargs)
        # This will accumulate the sum of squared errors
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        # This will accumulate the number of samples
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")


    def update(self: "RootMeanSquaredError", preds: Tensor, target: Tensor) -> None:
        """
        Accumulates squared errors and the number of samples for each batch.

        Parameters
        ----------
        preds : Tensor
            Predicted values. A tensor of shape `(N, ...)`, where `N` is the batch size.
        target : Tensor
            Ground truth values. A tensor of the same shape as `preds`.

        Notes
        -----
        This method is called once per batch to update the metric's state.
        """
        # Update the sum of squared errors and the number of samples
        squared_error = torch.sum((preds - target) ** 2)
        self.sum_squared_error += squared_error
        # target.numel() gives the number of elements in the target
        self.total += target.numel()


    def compute(self: "RootMeanSquaredError") -> Tensor:
        """
        Computes the Root Mean Squared Error (RMSE).

        Returns
        -------
        Tensor
            The RMSE, which is the square root of the mean of the accumulated squared errors.

        Notes
        -----
        This method is called after all batches have been processed, and it returns
        the final computed RMSE.
        """
        # Compute the RMSE by taking the square root of the average of the squared errors
        mean_squared_error = self.sum_squared_error / self.total
        return torch.sqrt(mean_squared_error)