from abc import ABC, abstractmethod
from typing import Optional


class BaseBackend:

    @abstractmethod
    def dispatch(self, model, train_fn, config_optim_fn, checkpoint: Optional[dict]):
        """
        Dispatches a training function to one or many processes on one or many compute nodes and does required setup
        """
        pass

    @abstractmethod
    def get_name(self):
        pass

    def prepare_data_loaders(self, train_loader, val_loader):
        return train_loader, val_loader

    @abstractmethod
    def train_step(self, train_fn):
        """
        Executes the train step given the user defined train step function.
        """
        pass

    @abstractmethod
    def val_step(self, val_fn):
        """
        Executes the val step given the user defined val step function.
        """
        pass

    @abstractmethod
    def optim_step(self, tensor):
        """
        Executes one backward pass, starting with the given tensor and calls step() on all optimizers.
        """
        pass

    @abstractmethod
    def scheduler_step(self, val_loss):
        pass

    @abstractmethod
    def to_device(self, data):
        """
        Moves a batch of data onto the designated device. `data` can be arbitrarily nested.
        """
        pass
