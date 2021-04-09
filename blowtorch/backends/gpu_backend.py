import copy
from typing import List
import os

import torch
from torch import cuda
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .base_backend import BaseBackend
from .apply_func import move_data_to_device
from ..utils import AMP_AVAILABLE
from .. import _writer as writer


class GPUBackend(BaseBackend):

    def __init__(self, num_nodes, num_gpus_per_node, node_rank, enable_amp):
        self.num_nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.node_rank = node_rank
        self.enable_amp = enable_amp

        self.use_ddp = num_nodes > 1 or num_gpus_per_node > 1
        self.model = None
        self.optimizers = {}
        self.schedulers = {}

        if self.use_ddp:
            writer.info(f'Using distributed data parallel across {self.num_nodes * self.num_gpus_per_node} '
                        f'GPUs on {self.num_nodes} nodes (this is node {self.node_rank})')

        if enable_amp and not AMP_AVAILABLE:
            raise ValueError('AMP is not supported by your PyTorch version, try torch>=1.6.')

    def dispatch(self, model, train_fn):
        if self.use_ddp:
            def train_fn_wrapper(process_index, args):
                # copy model for non-primary training processes
                _model = model if process_index == 0 and self.node_rank == 0 else copy.deepcopy(model)
                train_fn(_model, ...)
            mp.spawn(train_fn_wrapper, args, nprocs=self.num_gpus_per_node)
        else:
            train_fn(*args)

    def setup(self, model, config_optimizers_fn):
        if self.use_ddp:
            _init_ddp(rank, world_size)
            assert False  # todo destroy ddp

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

    def train_step(self, train_step_fn, **args):
        if self.enable_amp:
            with cuda.amp.autocast():
                return train_step_fn(**args)
        else:
            return train_step_fn(**args)

    def val_step(self, val_step_fn, **args):
        return val_step_fn(**args)

    def optim_step(self, tensor: torch.Tensor):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
            tensor.backward()
            optimizer.step()

    def set_optim_states(self, state_dicts):
        for name, state in state_dicts:
            self.optimizers[name].load_state_dict(state)

    def scheduler_step(self, val_loss):
        for scheduler in self.schedulers.values():
            scheduler.step(val_loss)

    def to_device(self, data):
        return move_data_to_device(data, torch.device(self.gpu_id))


def _init_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def _cleanup_ddp():
    dist.destroy_process_group()
