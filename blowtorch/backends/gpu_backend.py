import torch
from torch import cuda

from .base_backend import BaseBackend
from .apply_func import move_data_to_device
from ..utils import AMP_AVAILABLE


class GPUBackend(BaseBackend):

    def __init__(self, gpu_id, enable_amp=False):
        self.gpu_id = gpu_id
        self.enable_amp = enable_amp
        self.device = torch.device(f'cuda:{self.gpu_id}')
        self.model = None
        self.optimizers = {}
        self.schedulers = {}

        if not cuda.is_available() or not cuda.device_count() > self.gpu_id:
            raise ValueError(f'GPU [{self.gpu_id}] is not available.')

        if enable_amp and not AMP_AVAILABLE:
            raise ValueError('AMP is not supported by your PyTorch version, try torch>=1.6.')

    def setup(self, model, config_optimizers_fn):
        # move model parameters to specified GPU
        self.model = model.cuda(device=self.gpu_id)

        # setup AMP
        # writer.info('Using AMP with 16 bit precision')
        # TODO setup amp

        # setup optimizers for model parameters
        optimizer_config = config_optimizers_fn(model=self.model)
        self.optimizers = optimizer_config['optimizers']
        self.schedulers = optimizer_config['schedulers']
        if not isinstance(self.optimizers, dict):
            self.optimizers = {'main': self.optimizers}
        if not isinstance(self.schedulers, dict):
            self.schedulers = {'main': self.schedulers}

    def get_name(self):
        return f'GPUBackend[device={self.gpu_id}, AMP={self.enable_amp}]'

    def train_step(self, train_fn, **train_fn_args):
        if self.enable_amp:
            with cuda.amp.autocast():
                return train_fn(**train_fn_args)
        else:
            return train_fn(**train_fn_args)

    def val_step(self, val_fn, **val_fn_args):
        return val_fn(**val_fn_args)

    def optim_step(self, tensor: torch.Tensor):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
            tensor.backward()
            optimizer.step()

    def scheduler_step(self, val_loss):
        for scheduler in self.schedulers.values():
            scheduler.step(val_loss)

    def to_device(self, data):
        return move_data_to_device(data, torch.device(self.gpu_id))
