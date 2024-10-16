# TODO: Trainer with Hydra and Torch Lightning
# TODO: WandB logging ^_^
# TODO: Patience of 10
# TODO: Seed for reproducibility
# TODO: Checkpointing

#!/usr/bin/env python

import faulthandler
import logging
import torch

import hydra
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf

from fmstg.utils.logging import filter_device_available, get_logger, wandb_login, print_config, wandb_clean_config
from fmstg.utils.exceptions import print_exceptions
from fmstg.utils.seed import set_seed
from fmstg.config import instantiate_datamodule, instantiate_model, instantiate_task
from dotenv import load_dotenv

# Load dotenv files
load_dotenv()

# Log to traceback to stderr on segfault
faulthandler.enable(all_threads=False)

# If data loading is really not a bottleneck for you, uncomment this to silence the
# warning about it
# warnings.filterwarnings(
#     "ignore",
#     "The '\w+_dataloader' does not have many workers",
#     module="lightning",
# )
logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(
    filter_device_available
)


log = get_logger()


def get_callbacks(config):
    monitor = {"monitor": "val/top1", "mode": "max"}
    callbacks = [
        TQDMProgressBar(refresh_rate=1),
    ]
    if config.early_stopping is not None:
        stopper = EarlyStopping(
            patience=int(config.early_stopping),
            min_delta=0,
            strict=False,
            check_on_train_epoch_end=False,
            **monitor,
        )
        callbacks.append(stopper)
    return callbacks


@hydra.main(config_path="config", config_name="train", version_base=None)
@print_exceptions
def main(config: DictConfig):
    # rng = set_seed(config)

    # Resolve interpolations to work around a bug:
    # https://github.com/omry/omegaconf/issues/862
    OmegaConf.resolve(config)
    # wandb.init(**config.wandb, resume=(config.wandb.mode == "online") and "allow")
    wandb_login()
    print_config(config)

    log.info("Loading data")
    datamodule = instantiate_datamodule(config)
    datamodule.prepare_data()
    datamodule.setup("train")

    log.info("Instantiating model")

    # So for some reason we neeed to add to the configuration file the following:
    # - model.device: torch.device(cuda)
    # - model.A: adjacency_matrix of the dataset
    # The hack is
    # - device_name = "cuda"  => torch.device(device_name)
    # - adj_path = "file/path.npy" => np.load(adj_path)

    model = instantiate_model(config.model)

    task = instantiate_task(config, model, datamodule)

    # WANDB Logger
    wandb_logger = WandbLogger(
        project=config["wandb"]["project"],
        name=config["wandb"]["name"],
        config=wandb_clean_config(config),
        log_model="all",
    )
    # log_hyperparameters(logger, config, model)

    log.info("Instantiating trainer")
    callbacks = get_callbacks(config)
    trainer = Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=wandb_logger,
        plugins=[SLURMEnvironment(auto_requeue=False)],
    )

    log.info("Starting training!")
    trainer.fit(task, datamodule=datamodule)

    if config.eval_testset:
        log.info("Starting testing!")
        trainer.test(ckpt_path="best", datamodule=datamodule)

    wandb.finish()
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    best_score = trainer.checkpoint_callback.best_model_score
    return float(best_score) if best_score is not None else None


if __name__ == "__main__":
    main()
