from abc import ABC, abstractmethod


class BaseBackend:

    @abstractmethod
    def setup(self, model, config_optimizers_fn):
        """
        Initializes backend, moves model to device etc.
        Returns: dictionary of optimizers
        """
        pass

    @abstractmethod
    def get_name(self):
        pass

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
    def to_device(self, data):
        """
        Moves a batch of data onto the designated device. `data` can be arbitrarily nested.
        """
        pass
