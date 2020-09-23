from abc import ABC, abstractmethod

from torch import nn


class BaseLogger(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def log_config(self, config: dict):
        """
        Used to store the experiments' config, e.g. hyperparameters.
        """
        pass

    @abstractmethod
    def introduce_model(self, model: nn.Module):
        """
        Is called after model has been initialized and moved to the desired device. Can be used to e.g.
        print model statistics.
        """
        pass

    @abstractmethod
    def log_epoch(self, metrics: dict, epoch: int, is_validate: bool = False):
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint: dict, name: str):
        pass

    @abstractmethod
    def load_checkpoint(self, name: str):
        pass


class WandbLogger(BaseLogger):

    def __init__(self, **kwargs):
        super().__init__()
        import wandb
        self.wandb = wandb
        wandb.init(**kwargs)

    def log_config(self, config: dict):
        self.wandb.config = config

    def introduce_model(self, model: nn.Module):
        self.wandb.watch(model)

    def log_epoch(self, metrics: dict, epoch: int, is_validate: bool = False):
        prefix = 'val_' if is_validate else 'train_'
        self.wandb.log({(prefix + k): v for k, v in metrics.items()}, step=epoch)

    def save_checkpoint(self, checkpoint: dict, name: str):
        pass

    def load_checkpoint(self, name: str):
        pass
