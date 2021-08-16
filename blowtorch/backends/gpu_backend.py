from typing import Optional
import os

import torch
from torch import cuda
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from .base_backend import BaseBackend
from .apply_func import move_data_to_device
from ..utils import AMP_AVAILABLE, suppress
from ..ddp_utils import DistributedWrapper, replace_sampler, find_free_port, has_iterable_dataset
from .. import _writer as writer


class GPUBackend(BaseBackend):

    def __init__(
            self,
            num_nodes,
            num_gpus_per_node,
            node_rank,
            ddp_backend,
            ddp_find_unused_parameters,
            ddp_init_method,
            enable_amp
    ):
        self.num_nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.node_rank = node_rank
        self.ddp_backend = ddp_backend
        self.ddp_find_unused_parameters = ddp_find_unused_parameters
        self.ddp_init_method = ddp_init_method
        self.enable_amp = enable_amp

        self.use_ddp = num_nodes > 1 or num_gpus_per_node > 1
        self.world_size = num_nodes * num_gpus_per_node
        self.optimizers = {}
        self.schedulers = {}

        assert False  # todo seed (maybe global seed + rank),

        if enable_amp and not AMP_AVAILABLE:
            raise ValueError('AMP is not supported by your PyTorch version, try torch>=1.6.')

    def dispatch(self, model, train_fn, config_optim_fn, checkpoint: Optional[dict]):
        if self.use_ddp:
            def train_fn_wrapper(local_rank):
                rank = self.node_rank * self.num_gpus_per_node + local_rank
                with suppress(local_rank != 0):
                    with writer.task('Waiting for all nodes to join'):
                        self._init_ddp(rank)  # blocks
                    with writer.task('Setting up distributed environment'):
                        device = torch.device(f'cuda:{local_rank}')
                        _model = self._setup(model, device, config_optim_fn, checkpoint)
                        # free memory associated with checkpoint
                        del checkpoint
                        # wait for all processes
                        dist.barrier()

                # suppress stdout for all processes except rank 0
                with suppress(rank != 0):
                    train_fn(_model, rank)
                # TODO destroy ddp env (use desctructor handler in backend that is called from run on abort-c)
            mp.spawn(train_fn_wrapper, nprocs=self.num_gpus_per_node)
        else:
            device = torch.device('cuda:0')
            _model = self._setup(model, device, config_optim_fn, checkpoint)
            train_fn(_model, rank=0)

    def _init_ddp(self, rank):
        # if init method is env:// and the corresponding env variables are not set, assume local (single-node) training
        if self.ddp_init_method == 'env://':
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = 'localhost'
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = str(find_free_port())
        dist.init_process_group(self.ddp_backend, rank=rank, world_size=self.world_size,
                                init_method=self.ddp_init_method)

    def _setup(self, model, device, config_optim_fn, checkpoint: Optional[dict]):
        if checkpoint:
            model.load_state_dict(checkpoint['model'])
        # move model parameters to specified GPU
        torch.cuda.set_device(device)
        model.to(device)
        if self.use_ddp:
            # wrap in DistributedDataParallel
            model = DDP(model, device_ids=[device], find_unused_parameters=self.ddp_find_unused_parameters)

        # setup optimizers for model parameters
        optimizer_config = config_optim_fn(model)
        self.optimizers = optimizer_config['optimizers']
        self.schedulers = optimizer_config['schedulers']
        if not isinstance(self.optimizers, dict):
            self.optimizers = {'main': self.optimizers}
        if not isinstance(self.schedulers, dict):
            self.schedulers = {'main': self.schedulers}

        if checkpoint:
            self._set_optim_states(checkpoint['optimizers'])
        if self.use_ddp:
            # sync optimizer parameters from rank 0 to process group. Afterwards, all optimizers should remain
            # identical as they operate on the same model parameters and gradients
            self._sync_optim_states()

        return model

    def __repr__(self):
        return f'GPUBackend[{"distributed, " if self.num_nodes > 1 else ""}node {self.node_rank} ' \
               f'{"(main node) " if self.node_rank == 0 else ""}of {self.num_nodes},' \
               f' {self.num_gpus_per_node} GPUs per node{", using AMP" if self.enable_amp else ""}]'

    def prepare_data_loaders(self, train_loader, val_loader):
        if self.use_ddp:
            # wrap samplers in DistributedWrapper
            loaders = []
            for loader in (train_loader, val_loader):
                if has_iterable_dataset(loader):
                    raise NotImplementedError('IterableDataset currently not supported with DDP')
                if isinstance(loader.sampler, DistributedSampler):
                    raise ValueError('You\'re using a DistributedSampler with `ddp_set_samplers`=True. Either turn off'
                                     ' `ddp_set_samplers` and configure your DistributedSampler using the Run\'s '
                                     '`init` decorator or let blowtorch configure the appropriate samplers for you.')
                if loader.batch_sampler.batch_size % self.world_size != 0:
                    raise ValueError(f'Batch size {loader.batch_sampler.batch_size} should be divisible by the number'
                                     f'of processes {self.world_size}.')

                wrapped_sampler = DistributedWrapper(loader.sampler)
                loaders.append(replace_sampler(loader, wrapped_sampler, adjust_batch_size=True))
            return loaders

        # else pass through loaders
        return train_loader, val_loader

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

    def _set_optim_states(self, state_dicts):
        for name, state in state_dicts:
            self.optimizers[name].load_state_dict(state)

    def _sync_optim_states(self):
        keys = sorted(self.optimizers.keys())
        if dist.get_rank() == 0:
            broadcast_list = [self.optimizers[k].state_dict() for k in keys]
        else:
            broadcast_list = [None for _ in keys]
        dist.broadcast_object_list(broadcast_list, 0)
        self._set_optim_states({k: v for k, v in zip(keys, broadcast_list)})

    def scheduler_step(self, metrics):
        for scheduler in self.schedulers.values():
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics)
            else:
                scheduler.step()

    def to_device(self, data):
        return move_data_to_device(data, torch.device(self.gpu_id))
