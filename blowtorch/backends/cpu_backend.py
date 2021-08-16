from typing import Optional

import torch

from .base_backend import BaseBackend


class CPUBackend(BaseBackend):

    def __init__(self):
        self.optimizers = {}
        self.schedulers = {}

    def dispatch(self, model, train_fn, config_optim_fn, checkpoint: Optional[dict] = None):
        model.cpu()

        if checkpoint:
            model.load_state_dict(checkpoint['model'])

        # setup optimizers for model parameters
        optimizer_config = config_optim_fn(model=model)
        if isinstance(optimizer_config, tuple):
            self.optimizers, self.schedulers = optimizer_config
        else:
            self.optimizers, self.schedulers = optimizer_config, {}
        if not isinstance(self.optimizers, dict):
            self.optimizers = {'main': self.optimizers}
        if not isinstance(self.schedulers, dict):
            self.schedulers = {'main': self.schedulers}

        if checkpoint:
            self._set_optim_states(checkpoint['optimizers'])
            self._set_scheduler_states(checkpoint['schedulers'])

        train_fn(model, rank=0)

    def __repr__(self):
        return f'CPUBackend'

    def train_step(self, train_fn, **train_fn_args):
        return train_fn(**train_fn_args)

    def val_step(self, val_fn, **val_fn_args):
        return val_fn(**val_fn_args)

    def optim_step(self, tensor):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
            tensor.backward()
            optimizer.step()

    def scheduler_step(self, metrics):
        for scheduler in self.schedulers.values():
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics)
            else:
                scheduler.step()

    def to_device(self, data):
        return data

    def _set_optim_states(self, state_dicts):
        for name, state in state_dicts:
            self.optimizers[name].load_state_dict(state)

    def _set_scheduler_states(self, state_dicts):
        for name, state in state_dicts:
            self.schedulers[name].load_state_dict(state)