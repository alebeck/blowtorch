from typing import Optional, List
from pathlib import Path
from datetime import datetime
import functools

import yaml
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader

from .utils import make_wrapper, get_terminal_writer
from .config import TrainingConfig
from .bound_functions import BoundFunctions
from .apply_func import move_data_to_device  # todo hook
from .loggers import BaseLogger


class Run:

    def __init__(self, config_files: Optional[List]):
        self._bound_functions = BoundFunctions()
        self._config = None
        self._model = None
        self._train_set = None
        self._val_set = None
        self._batch_size = None
        self._num_workers = None
        self._max_epochs = None
        self._use_gpu = None
        self._gpu_id = None
        self._resume = None
        self._optimize_metric = None
        self._checkpoint_metric = None
        self._smaller_is_better = None  # TODO state which to minimize/checkpoint on in result dict
        self._collate_fn = None
        self._optimizers = None
        self._is_validate = None
        self._config_files = []
        self.writer = None

        # TODO support run() args in config
        if config_files is None:
            config_files = []
        for file in config_files:
            self.add_config(file)
        self._config = TrainingConfig(self._config_files)

    def _init_data(self):
        if isinstance(self._batch_size, tuple):
            batch_size_train, batch_size_val = self._batch_size
        else:
            batch_size_train, batch_size_val = self._batch_size, self._batch_size

        train_loader = DataLoader(self._train_set, batch_size_train, num_workers=self._num_workers,
                                  shuffle=True, collate_fn=self._collate_fn)
        val_loader = DataLoader(self._val_set, batch_size_val, num_workers=self._num_workers, shuffle=False,
                                collate_fn=self._collate_fn)
        return train_loader, val_loader

    # TODO types, docstrings
    # TODO pin_memory
    # todo clear cache before start
    # todo hooks
    # todo cleanup code (extra files for optim, devices etc.)
    # todo save on ctrl-C
    # todo look at pl GPUbackend (amp optimizuation etc)
    def run(self,
            model: torch.nn.Module,
            train_set: Dataset,
            val_set: Dataset,
            logger: BaseLogger,
            batch_size=1,
            num_workers=0,
            max_epochs=1,
            use_gpu=True,
            gpu_id=0,
            resume=False,
            optimize_metric=None,
            checkpoint_metric=None,
            smaller_is_better=True,
            collate_fn=None
            ):
        self._model = model
        self._train_set = train_set
        self._val_set = val_set
        self._logger = logger
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._max_epochs = max_epochs
        self._use_gpu = use_gpu
        self._gpu_id = gpu_id
        self._resume = resume
        self._optimize_metric = optimize_metric
        self._checkpoint_metric = checkpoint_metric
        self._smaller_is_better = smaller_is_better
        self._collate_fn = collate_fn

        self.writer = get_terminal_writer()

        self._logger.log_config(self._config.get_raw_config())

        log_path = Path('training_logs') / datetime.now().strftime("%y-%m-%d_%H-%M-%S")

        if 'train_step' in self._bound_functions and 'val_step' in self._bound_functions:
            mode = 'step'
        elif 'train_epoch' in self._bound_functions and 'val_epoch' in self._bound_functions:
            mode = 'epoch'
        else:
            err_str = 'Need to register either both train_step/val_step or both train_epoch/val_epoch functions.'
            self.writer.error(err_str)
            raise ValueError(err_str)

        start_epoch = 0

        # load checkpoint
        if self._resume:
            with self.writer.task('Loading checkpoint'):
                checkpoint = torch.load(self._resume)
                self._model.load_state_dict(checkpoint['model'])
                start_epoch = checkpoint['epoch']

        train_loader, val_loader = self._init_data()

        # GPU config
        if not torch.cuda.device_count() > self._gpu_id:
            err_str = f'GPU [{self._gpu_id}] is not available.'
            self.writer.error(err_str)
            raise ValueError(err_str)

        self._use_gpu = self._use_gpu and cuda.is_available()
        if self._use_gpu:
            self._model.cuda(device=self._gpu_id)
            self.writer.info(f'Using GPU [{self._gpu_id}]')

        self._logger.introduce_model(self._model)

        self._optimizers = self._bound_functions['configure_optimizers'](model=self._model)
        if not isinstance(self._optimizers, dict):
            self._optimizers = {'main': self._optimizers}

        if self._resume:
            self.writer.info(f'Resuming training from checkpoint {self._resume}')
            # resume optimizer state
            self._set_optim_states(checkpoint['optimizers'])

        best_val = float('inf') if self._smaller_is_better else 0.
        best_epoch = None

        for epoch in range(start_epoch, start_epoch + self._max_epochs):
            metrics = {}  # stores metrics of current epoch

            # train
            self._model.train()

            with self.writer.task(f'Training epoch {epoch}') as t:
                if mode == 'step':
                    step_metrics = []
                    for batch in t.tqdm(train_loader):
                        if self._use_gpu:
                            batch = move_data_to_device(batch, torch.device(self._gpu_id))

                        train_metrics = self._bound_functions['train_step'](batch=batch, model=self._model,
                                                                            is_validate=False)
                        assert isinstance(train_metrics, dict), '"train_step" should return a metrics dict.'

                        if epoch > 0:
                            self._optim_step(train_metrics)

                        step_metrics.append({k: float(v) for k, v in train_metrics.items()})

                    # calculate mean metrics
                    metrics['train'] = {
                        metric: np.array([dic[metric] for dic in step_metrics]).mean() for metric in step_metrics[0]
                    }

                elif mode == 'epoch':
                    train_metrics = self._bound_functions['train_epoch'](data_loader=train_loader, model=self._model,
                                                                         is_validate=False, optimizers=self._optimizers)
                    assert isinstance(train_metrics, dict), '"train_epoch" should return a metrics dict.'
                    metrics['train'] = {k: float(v) for k, v in train_metrics.items()}

                self._logger.log_epoch(metrics['train'], epoch, is_validate=False)

                status_str = f'[Epoch {epoch} / Train] ' \
                             + ' '.join([f'{k}: {self._round(v)}' for k, v in metrics['train'].items()])
                t.success(status_str)

            # val
            self._model.eval()

            with self.writer.task(f'Validating epoch {epoch}') as t:
                if mode == 'step':
                    step_metrics = []
                    for batch in t.tqdm(val_loader):
                        if self._use_gpu:
                            batch = move_data_to_device(batch, torch.device(self._gpu_id))

                        val_metrics = self._bound_functions['val_step'](batch=batch, model=self._model,
                                                                        is_validate=True)
                        assert isinstance(val_metrics, dict), '"val_step" should return a metrics dict.'
                        step_metrics.append({k: float(v) for k, v in val_metrics.items()})

                    # calculate mean metrics
                    # todo agg function
                    metrics['val'] = {
                        metric: np.array([dic[metric] for dic in step_metrics]).mean() for metric in step_metrics[0]
                    }

                elif mode == 'epoch':
                    val_metrics = self._bound_functions['val_epoch'](data_loader=val_loader, model=self._model,
                                                                     is_validate=True, optimizers=self._optimizers)
                    assert isinstance(val_metrics, dict), '"val_epoch" should return a metrics dict.'
                    metrics['val'] = {k: float(v) for k, v in val_metrics.items()}

                self._logger.log_epoch(metrics['val'], epoch, is_validate=True)

                status_str = f'[Epoch {epoch} / Val]   ' \
                             + ' '.join([f'{k}: {self._round(v)}' for k, v in metrics['val'].items()])
                t.success(status_str)

            if self._checkpoint_metric is None:
                metric = list(val_metrics.keys())[0]
                self.writer.info(f'Using "{metric}" as checkpointing metric')
                self._checkpoint_metric = metric

            # save checkpoint
            if (self._smaller_is_better and metrics['val'][self._checkpoint_metric] < best_val) or \
                    (not self._smaller_is_better and metrics['val'][self._checkpoint_metric] > best_val):

                with self.writer.task(f'Saving checkpoint at epoch {epoch}'):
                    checkpoint = {
                        'model': self._model.state_dict(),
                        'optimizers': {name: optim.state_dict() for name, optim in self._optimizers.items()},
                        'epoch': epoch + 1
                    }
                    checkpoint_path = log_path / 'checkpoints' / f'epoch_{epoch}.pt'
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, checkpoint_path)
                    # delete old checkpoint
                    (log_path / 'checkpoints' / f'epoch_{best_epoch}.pt').unlink(missing_ok=True)

                best_val = metrics['val'][self._checkpoint_metric]
                best_epoch = epoch

        self.writer.success(f'Training finished')

    def _optim_step(self, metrics):
        if self._optimize_metric is None:
            metric = list(metrics.keys())[0]
            self.writer.info(f'Selected metric "{metric}" for minimization')
            self._optimize_metric = metric

        for optimizer in self._optimizers.values():
            optimizer.zero_grad()
            metrics[self._optimize_metric].backward()
            optimizer.step()

    def add_config(self, path):
        self._config_files.append(path)

    @functools.wraps(run)
    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    def __getitem__(self, item):
        return self._config[item]

    # DECORATORS #

    # TODO docstrings

    def init(self, f):
        self._bound_functions['init'] = f
        return make_wrapper(f)

    def train_step(self, f):
        self._bound_functions['train_step'] = f
        return make_wrapper(f)

    def validate_step(self, f):
        self._bound_functions['val_step'] = f
        return make_wrapper(f)

    def train_epoch(self, f):
        self._bound_functions['train_epoch'] = f
        return make_wrapper(f)

    def validate_epoch(self, f):
        self._bound_functions['val_epoch'] = f
        return make_wrapper(f)

    def configure_optimizers(self, f):
        self._bound_functions['configure_optimizers'] = f
        return make_wrapper(f)

    def _set_optim_states(self, state_dicts):
        for name, state in state_dicts:
            self._optimizers[name].load_state_dict(state)

    def _round(self, value):
        return round(value, 4)
