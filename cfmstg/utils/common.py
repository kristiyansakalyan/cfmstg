"""Common Utils"""

from typing import Literal

import torch

EvaluationMode = Literal["val", "test"]


def gather(consts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Gathers values from the last dimension of `consts` at the indices specified by `t`,
    then reshapes the output to a 4D tensor.

    Parameters
    ----------
    consts : torch.Tensor
        A tensor of shape (..., N) containing values from which to gather. `N` is the number of steps (e.g., timesteps).
    t : torch.Tensor
        A tensor of indices representing the timesteps to gather from `consts`.

    Returns
    -------
    torch.Tensor
        A tensor of gathered values reshaped to shape (-1, 1, 1, 1).
    """
    # Gather values from consts at indices t
    c = consts.gather(-1, t)

    # Reshape the gathered tensor to shape (-1, 1, 1, 1)
    return c.reshape(-1, 1, 1, 1)
