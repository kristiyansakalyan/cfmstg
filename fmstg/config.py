"""Configuration file"""
from omegaconf import DictConfig

from fmstg.data.datamodule import TrafficDataModule
from fmstg.model.diffstg import DiffSTG
from fmstg.task.forecasting import ForecastingModel, ForecastingTask


def instantiate_datamodule(config: DictConfig):
    if config.data.name == "PEMS":
        return TrafficDataModule(config)


def instantiate_model(config: DictConfig):
    if config.name == "diffstg":
        return DiffSTG(config)


def instantiate_task(
    config: DictConfig, model: ForecastingModel, datamodule: TrafficDataModule
):
    if config.task.name == "forecasting":
        return ForecastingTask(model, datamodule, config)
