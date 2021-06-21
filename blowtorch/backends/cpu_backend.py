import torch

from .base_backend import BaseBackend


class CPUBackend(BaseBackend):

    def setup(self, model, config_optimizers_fn, checkpoint):
        # setup optimizers for model parameters and return
        return config_optimizers_fn(model=model)

    def get_name(self):
        return f'CPUBackend'

    def train_step(self, train_fn, **train_fn_args):
        return train_fn(**train_fn_args)

    def val_step(self, val_fn, **val_fn_args):
        return val_fn(**val_fn_args)

    def optim_step(self, tensor):
        raise NotImplementedError()

    def scheduler_step(self, metrics):
        for scheduler in self.schedulers.values():
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics)
            else:
                scheduler.step()

    def to_device(self, data):
        return data
