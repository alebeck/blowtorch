import torch

from .base_backend import BaseBackend
from .apply_func import move_data_to_device


class GPUBackend(BaseBackend):

    def __init__(self, gpu_id):
        self.gpu_id = gpu_id

    def setup(self, model, config_optimizers_fn):
        # move model parameters to specified GPU
        model.cuda(device=self.gpu_id)

        # setup optimizers for model parameters and return
        return config_optimizers_fn(model=model)

    def get_name(self):
        return f'GPUBackend[device={self.gpu_id}]'

    def train_step(self, train_fn, **train_fn_args):
        return train_fn(**train_fn_args)

    def val_step(self, val_fn, **val_fn_args):
        return val_fn(**val_fn_args)

    def to_device(self, data):
        return move_data_to_device(data, torch.device(self.gpu_id))
