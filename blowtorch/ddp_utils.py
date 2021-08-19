import socket
from contextlib import closing
import math
from typing import Optional, Iterator

import inspect
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, BatchSampler, Sampler


def cleanup_ddp():
    dist.destroy_process_group()


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def has_iterable_dataset(dataloader: torch.utils.data.DataLoader):
    return hasattr(dataloader, 'dataset') and isinstance(dataloader.dataset, torch.utils.data.IterableDataset)


# The following is adapted from Pytorch Lightning (https://github.com/PyTorchLightning/pytorch-lightning/blob/
# 71b4611c64059d7589e4d80115209fd2c89e8bdb/pytorch_lightning/trainer/data_loading.py#L132)
def _resolve_batch_sampler(dl_args, dataloader, sampler, adjust_batch_size):
    world_size = dist.get_world_size()
    batch_sampler = getattr(dataloader, "batch_sampler")
    if batch_sampler is not None and type(batch_sampler) is not BatchSampler:
        batch_sampler = type(batch_sampler)(
            sampler,
            batch_size=batch_sampler.batch_size // world_size if adjust_batch_size else batch_sampler.batch_size,
            drop_last=batch_sampler.drop_last,
        )
        dl_args['batch_sampler'] = batch_sampler
        dl_args['batch_size'] = 1
        dl_args['shuffle'] = False
        dl_args['sampler'] = None
        dl_args['drop_last'] = False
    else:
        dl_args['sampler'] = sampler
        dl_args['shuffle'] = False
        dl_args['batch_sampler'] = None
        if adjust_batch_size:
            dl_args['batch_size'] = dl_args['batch_size'] // world_size

    return dl_args


def replace_sampler(dataloader, sampler, adjust_batch_size):
    skip_keys = ('sampler', 'batch_sampler', 'dataset_kind')
    skip_signature_keys = ('args', 'kwargs', 'self')

    attrs = {k: v for k, v in vars(dataloader).items() if not k.startswith("_")}

    params = set(inspect.signature(dataloader.__init__).parameters)
    contains_dataset = True

    if type(dataloader) is not DataLoader:
        contains_dataset = "dataset" in params
        params.update(inspect.signature(DataLoader.__init__).parameters)

    dl_args = {name: attrs[name] for name in params if name in attrs and name not in skip_keys}

    dl_args = _resolve_batch_sampler(dl_args, dataloader, sampler, adjust_batch_size)

    multiprocessing_context = dataloader.multiprocessing_context
    dl_args['multiprocessing_context'] = multiprocessing_context

    missing_kwargs = params.difference(skip_signature_keys).difference(dl_args)
    if missing_kwargs:
        """
        Example:
        class CustomDataLoader(DataLoader):
            def __init__(self, num_features, dataset, *args, **kwargs):
                self.num_features = num_features
                super().__init__(dataset, *args, **kwargs)
        """
        dataloader_cls_name = dataloader.__class__.__name__
        raise ValueError(
            f"Trying to inject DistributedSampler within {dataloader_cls_name} class."
            "This would fail as your DataLoader doesn't expose all its __init__ parameters as attributes. "
            f"Missing attributes are {missing_kwargs}. "
            f"HINT: If you wrote the {dataloader_cls_name} class, add the `__init__` arguments as attributes or ",
            "manually add DistributedSampler as "
            f"{dataloader_cls_name}(dataset, ..., sampler=DistributedSampler(dataset, ...)).",
        )

    if not contains_dataset:
        dl_args.pop('dataset')

    dataloader = type(dataloader)(**dl_args)
    dataloader.multiprocessing_context = multiprocessing_context
    return dataloader


class DistributedWrapper(Sampler):

    def __init__(self, sampler: Sampler, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, drop_last: bool = False) -> None:
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.is_main = rank == 0

        # If the sampler length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the sampler will be split equally.
        if self.drop_last and len(self.sampler) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.sampler) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.sampler) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator:
        # broadcast indices from main process to workers
        indices = list(self.sampler)
        dist.broadcast_object_list(indices, 0)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
