import inspect
import logging
import os
import re

import omegaconf
import rich
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from rich.syntax import Syntax

import wandb

DEVICE_AVAILABLE = re.compile("[TIH]PU available: ")


def filter_device_available(record):
    """Filter the availability report for all the devices we don't have."""
    return not DEVICE_AVAILABLE.match(record.msg)


def get_logger() -> logging.Logger:
    """Gets the logger for the current module"""
    caller = inspect.stack()[1]
    module = inspect.getmodule(caller.frame)
    logger_name = None
    if module is not None:
        logger_name = module.__name__.split(".")[-1]
    return logging.getLogger(logger_name)


def wandb_login() -> None:
    """WandB login"""
    wandb_key = os.getenv("WANDB")

    if not wandb_key:
        raise ValueError("Could not load WandB Key")

    wandb.login(key=wandb_key, relogin=True)

def wandb_clean_config(config: DictConfig) -> dict:
    """
    Recursively cleans the config dictionary by converting any list values
    to strings while keeping the other values unchanged.

    Parameters
    ----------
    config : dict
        The configuration dictionary to clean.

    Returns
    -------
    dict
        A new dictionary where all list values are converted to strings.
    """
    # Just a hack to log config to WandB that does not support lists.
    def clean(item):
        """Recursively cleans individual items in the config."""
        if isinstance(item, DictConfig):
            # If it's a dictionary, recursively clean each value
            return {k: clean(v) for k, v in item.items()}
        elif isinstance(item, omegaconf.listconfig.ListConfig):
            # If it's a list, convert it to a string
            return omegaconf.StringNode(str(item))
        else:
            # Otherwise, return the item unchanged
            return item

    return clean(config)


@rank_zero_only
def print_config(config: DictConfig) -> None:
    content = OmegaConf.to_yaml(config, resolve=True)
    rich.print(Syntax(content, "yaml"))
