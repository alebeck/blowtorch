from collections import defaultdict
from datetime import datetime
from typing import Optional, List, Union
from pathlib import Path
import random
import functools
import warnings
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from coolname import generate_slug, replace_random

from . import _writer as writer
from .backends.cpu_backend import CPUBackend
from .backends.gpu_backend import GPUBackend
from .bound_functions import call
from .utils import get_highest_run, std_round, seed_all, set_deterministic
from .config import TrainingConfig
from .bound_functions import BoundFunctions
from .loggers import BaseLogger, LoggerSet, StandardLogger


class Run:
    """
    Represents an individual training run.
    """

    def __init__(self, config_files: Optional[List] = None, random_seed: int = None):
        self._bound_functions = BoundFunctions()
        self._config = None
        self._backend = None
        self._logger = None
        self.train_loader = None
        self.val_loader = None
        self._loggers = None
        self._num_epochs = None
        self._use_gpu = None
        self._resume_checkpoint = None
        self._save_path = None
        self.checkpoints_path = None
        self._run_name = None
        self._val_every = None
        self._val_at = None
        self._optimize_metric = None
        self._checkpoint_metric = None
        self._checkpoint_every = None
        self._smaller_is_better = None  # TODO state which to minimize/checkpoint on in result dict
        self._optimize_first = None
        self._enable_amp = None
        self._detect_anomalies = None
        self._is_validate = None
        self._start_epoch = 0
        self._is_main_node = None

        self._train_loader = None
        self._val_loaders = []
        self._config = TrainingConfig([] if config_files is None else config_files)

        self.random_seed = random_seed
        if random_seed:
            seed_all(random_seed)

    # TODO types, docstrings
    # TODO pin_memory
    # todo clear cache before start
    # todo hooks
    # todo cleanup code (extra files for optim, devices etc.)
    # todo save on ctrl-C
    # todo look at pl GPUbackend (amp optimizuation etc)
    def run(self,
            model: torch.nn.Module,
            *,
            loggers: Optional[List[BaseLogger]] = None,
            num_epochs=10,
            use_gpu=True,
            num_nodes=1,
            num_gpus_per_node=1,
            node_rank=0,
            ddp_backend='nccl',
            ddp_init_method='env://',
            ddp_find_unused_parameters=False,
            resume_checkpoint: Optional[Union[str, Path]] = None,
            save_path='train_logs',
            run_name=None,
            optimize_metric=None,
            checkpoint_metric=None,
            checkpoint_every=1,
            smaller_is_better=True,
            optimize_first=False,
            enable_amp=False,
            detect_anomalies=False
            ):
        """
        Starts the training run.

        Args:
            model:
            loggers: list of loggers that subscribe to various logging events
            num_epochs:
            use_gpu:
            num_nodes:
            num_gpus_per_node:
            node_rank: when num_nodes > 1, this specifies the ordinal number of the current node within all nodes
            ddp_backend:
            ddp_init_method:
            ddp_find_unused_parameters:
            resume_checkpoint: path to checkpoint to resume training from
            save_path: path to directory that blowtorch will save logs and checkpoints to
            run_name: name associated with this run, will be randomly created if None
            optimize_metric: train metric that will be used for optimization, will pick the first returned one if None
            checkpoint_metric: validation metric that will be used for checkpointing, will pick the first returned one
            if None
            checkpoint_every: every checkpoint_every epochs a checkpoint is saved, disregarding performance of the
            current model. This way it's always possible to resume the run from the latest (or near-latest) state
            smaller_is_better: ``True`` if we want to minimize, ``False`` if maximize
            optimize_first: whether optimization should occur during the first epoch
            enable_amp:
            detect_anomalies: enable autograd anomaly detection
        """
        self._loggers = loggers
        self._num_epochs= num_epochs
        self._use_gpu = use_gpu
        self._resume_checkpoint = resume_checkpoint
        self._run_name = run_name
        self._save_path = save_path
        self._optimize_metric = optimize_metric
        self._checkpoint_metric = checkpoint_metric
        self._checkpoint_every = checkpoint_every
        self._smaller_is_better = smaller_is_better
        self._optimize_first = optimize_first
        self._enable_amp = enable_amp
        self._detect_anomalies = detect_anomalies

        self._save_path = Path(self._save_path)
        self._save_path.mkdir(parents=True, exist_ok=True)

        self._is_main_node = num_nodes == 1 or node_rank == 0

        # assign new random.Random() instance to coolname, such that slugs are different even though we have seeded
        replace_random(random.Random())

        if self._resume_checkpoint:
            if self._run_name is not None:
                raise ValueError('A run name cannot be specified when resuming from a previous run.')
            self._resume_checkpoint = Path(self._resume_checkpoint)
            if not self._resume_checkpoint.is_dir() or not (self._resume_checkpoint / 'checkpoints').exists():
                raise ValueError('Path to resume from should be the parent directory of the "checkpoints" folder.')
            self._run_name = self._resume_checkpoint.stem.split('_')[-1]
            self._save_path = self._resume_checkpoint
        else:
            if self._run_name is None:
                self._run_name = generate_slug(2)
                assert '_' not in self._run_name
            elif '_' in self._run_name:
                raise ValueError('Run name cannot contain "_".')
            # append consecutive number to run name
            self._run_name += f'-{get_highest_run(self._save_path) + 1}'
            self._save_path = self._save_path / (datetime.now().strftime("%y-%m-%d_%H-%M-%S") + '_' + self._run_name)
            self._save_path.mkdir(parents=True, exist_ok=False)

        self.checkpoints_path = self._save_path / 'checkpoints'
        self.checkpoints_path.mkdir(exist_ok=True)

        # backend initialization
        try:
            if self._use_gpu:
                self._backend = GPUBackend(num_nodes, num_gpus_per_node, node_rank, ddp_backend,
                                           ddp_find_unused_parameters, ddp_init_method, enable_amp)
            else:
                self._backend = CPUBackend()
        except Exception as e:
            writer.error(str(e))
            raise

        writer.info(f'Using {self._backend}')

        checkpoint = None
        if self._is_main_node:
            if self._resume_checkpoint:
                # only need to pass weights on main process, for it is distributed to the other nodes automatically
                writer.info(f'Resuming training from checkpoint {self._resume_checkpoint}')
                checkpoint = torch.load(self.checkpoints_path / 'latest', map_location='cpu')
                self._start_epoch = checkpoint['epoch']

            if not self._optimize_first and self._start_epoch == 0:
                writer.info('Not optimizing during first epoch')

        self._backend.dispatch(model, self._train_fn, self._bound_functions['configure_optimizers'], checkpoint)

    def _train_fn(self, model, rank):
        if self.random_seed:
            # we want every training process to have a different, but deterministic random seed
            seed_all(self.random_seed + rank)

        is_main = rank == 0
        best_val = float('inf') if self._smaller_is_better else float('-inf')
        did_warn_train_metrics = False

        self._init_loggers(is_main)
        self._logger.before_training_start(self._config.get_raw_config(), model, self._bound_functions)

        # give backend the chance to wrap dataloaders, e.g. with samplers for multi-process training
        train_loader, *val_loaders = self._backend.prepare_data_loaders(self._train_loader, *self._val_loaders)

        for epoch in range(self._start_epoch, self._start_epoch + self._num_epochs):

            # ===== TRAINING ==== #

            model.train()
            torch.set_grad_enabled(True)
            train_metrics = defaultdict(float)

            with writer.task(f'Training epoch {epoch}') as t:
                for batch in t.tqdm(train_loader):
                    batch = self._backend.to_device(batch)

                    with torch.autograd.set_detect_anomaly(self._detect_anomalies) if is_main else nullcontext():
                        # don't calculate grads if we're in epoch zero and not optimizing
                        torch.set_grad_enabled(self._optimize_first or epoch > 0)

                        with self._backend.get_train_step_context():
                            metrics = call(self._bound_functions['train_step'], batch=batch, model=model,
                                           is_validate=False, device=self._backend.device, epoch=epoch)

                        if not isinstance(metrics, dict):
                            if not did_warn_train_metrics:
                                writer.warning('Received a single return value from `train_step`, assuming '
                                               '"loss". Return a dict to explicitly name the metric(s).')
                                did_warn_train_metrics = True
                            metrics = {'loss': metrics}

                        if self._optimize_metric is None:
                            metric = list(metrics.keys())[0]  # TODO possibility to state which one to optimize
                            writer.info(f'Selected metric "{metric}" for minimization')
                            self._optimize_metric = metric

                        if self._optimize_first or epoch > 0:
                            self._backend.optim_step(metrics[self._optimize_metric])

                    for key in metrics:
                        train_metrics[key] += float(metrics[key]) / len(train_loader)

                    t.set_current_metrics({
                        self._optimize_metric: std_round(metrics[self._optimize_metric].item())
                    })

                # give backend the possibility to synchronize metrics across multiple processes (blocking)
                self._backend.synchronize_metrics(train_metrics)

                if 'after_train' in self._bound_functions:
                    call(self._bound_functions['after_train'], metrics=train_metrics, model=model,
                         device=self._backend.device, epoch=epoch)
                    for m in train_metrics:
                        try:
                            train_metrics[m] = float(train_metrics[m])
                        except TypeError:
                            pass

                self._logger.after_pass(metrics=train_metrics, charts=None, epoch=epoch, is_validate=False)

                t.success(f'[Epoch {epoch} / Train] ' +
                          ' '.join([f'{k}: {std_round(v)}' for k, v in train_metrics.items() if isinstance(v, float)]))

            # ===== VALIDATION ==== #

            if (self._val_every is None or epoch % self._val_every == 0) and epoch >= self._val_at:

                model.eval()
                torch.set_grad_enabled(False)
                val_metrics_list = []
                val_charts_dict = {}

                with writer.task(f'Validating epoch {epoch}') as t:
                    for val_loader, val_func in zip(val_loaders, self._bound_functions['val_step']):
                        val_metrics = defaultdict(float)
                        for batch_index, batch in enumerate(t.tqdm(val_loader)):
                            batch = self._backend.to_device(batch)

                            with self._backend.get_val_step_context():
                                metrics, charts = call(val_func, batch=batch, model=model, is_validate=True,
                                           device=self._backend.device, epoch=epoch, batch_index=batch_index)

                            if not isinstance(metrics, dict):
                                metrics = {'loss': metrics}

                            for key in metrics:
                                val_metrics[key] += float(metrics[key]) / len(val_loader)

                            display_metric = self._optimize_metric if self._optimize_metric in metrics else \
                                list(metrics.keys())[0]
                            t.set_current_metrics({display_metric: std_round(metrics[display_metric].item())})
                            
                            if charts is not None and isinstance(charts, dict):
                                for k, v in charts.items():
                                    if v is None:
                                        continue
                                    val_charts_dict[k] = v

                        self._backend.synchronize_metrics(val_metrics)

                        if 'after_val' in self._bound_functions:
                            call(self._bound_functions['after_val'], metrics=val_metrics, model=model,
                                 device=self._backend.device, epoch=epoch)
                            for m in val_metrics:
                                try:
                                    val_metrics[m] = float(val_metrics[m])
                                except TypeError:
                                    pass

                        val_metrics_list.append(val_metrics)

                    # join val_metrics dicts
                    val_metrics = {k: v for d in val_metrics_list for k, v in d.items()}

                    # TODO specify metric to do scheduling on
                    # if self._optimize_first is False, a warning will be raised by the schedulers which suggests that
                    # optim.step() is called after scheduler.step(), which would normally result in the first epoch being
                    # skipped from the learning rate scheduler. In our case, however, optim.step() was not called because
                    # of self._optimize_first is False, and the epoch counter should indeed be increased.
                    if epoch == 0 and not self._optimize_first:
                        warnings.simplefilter(action='ignore', category=UserWarning)
                        self._backend.scheduler_step(val_metrics[self._optimize_metric])
                        warnings.filterwarnings('default')
                    else:
                        self._backend.scheduler_step(val_metrics[self._optimize_metric])
                    
                    self._logger.after_pass(metrics=val_metrics, charts=val_charts_dict, epoch=epoch, is_validate=True) 

                    t.success(f'[Epoch {epoch} / Val]   ' +
                          ' '.join([f'{k}: {std_round(v)}' for k, v in val_metrics.items() if isinstance(v, float)]))

                if self._checkpoint_metric is None:
                    metric = list(val_metrics.keys())[0]
                    writer.info(f'Selected metric "{metric}" for checkpointing')
                    self._checkpoint_metric = metric

                is_best = (self._smaller_is_better and val_metrics[self._checkpoint_metric] < best_val) or \
                      (not self._smaller_is_better and val_metrics[self._checkpoint_metric] > best_val)
            else:
                is_best = False

            if is_main and (is_best or epoch % self._checkpoint_every == 0):
                with writer.task(f'Saving checkpoint'):
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizers': {name: optim.state_dict() for name, optim in self._backend.optimizers.items()},
                        'schedulers': {name: sched.state_dict() for name, sched in self._backend.schedulers.items()},
                        'next_epoch': epoch + 1
                    }
                    path = self.checkpoints_path / f'epoch_{epoch}.pt'
                    torch.save(checkpoint, path)

                    latest_path = self.checkpoints_path / 'latest'
                    best_path = self.checkpoints_path / 'best'

                    if latest_path.is_symlink():
                        # delete previous latest checkpoint
                        checkpoint_file = latest_path.resolve()
                        latest_path.unlink()
                        if not (best_path.is_symlink() and best_path.resolve() == checkpoint_file):
                            # best_path symlink does not link to this checkpoint, so we can delete it
                            checkpoint_file.unlink()
                    # create new latest symlink
                    latest_path.symlink_to(path.name)

                    if is_best:
                        # delete old best checkpoint and symlink new one
                        if best_path.is_symlink():
                            checkpoint_file = best_path.resolve()
                            best_path.unlink()
                            checkpoint_file.unlink()
                        best_path.symlink_to(path.name)

                best_val = val_metrics[self._checkpoint_metric]

        writer.success(f'Training finished')

    def _init_loggers(self, is_main):
        if self._loggers is None:
            self._loggers = []
        if not isinstance(self._loggers, (list, tuple)):
            self._loggers = [self._loggers]

        if is_main:
            self._logger = LoggerSet([StandardLogger()] + self._loggers)
        else:
            self._logger = LoggerSet([])

        self._logger.setup(self._save_path, self._run_name, self._resume_checkpoint is not None)

    def get_raw_config(self):
        return self._config.get_raw_config()

    def __getitem__(self, item):
        return self._config[item]

    @staticmethod
    def seed_all(seed):
        seed_all(seed)

    @staticmethod
    def set_deterministic(deterministic):
        set_deterministic(deterministic)

    @functools.wraps(run)
    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    # DECORATORS #

    # TODO docstrings

    def configure_optimizers(self, f):
        self._bound_functions['configure_optimizers'] = f
        return f

    def train_step(self, data_loader):
        def decorator(f):
            self._bound_functions['train_step'] = f
            self._train_loader = data_loader
            return f
        return decorator

    def validate_step(self, data_loader, every=1, at=0):
        def decorator(f):
            self._bound_functions['val_step'] = f
            self._val_loaders.append(data_loader)
            self._val_every = every
            self._val_at = at
            return f
        return decorator

    # hooks
    def after_train(self, f):
        """
        Registers a hook called after every training pass. Supported parameters:
        `metrics` - dict of this epoch's synchronized training metrics, can be mutated
        `model` - the model
        `device` - device that the model/data resides on
        `epoch` - current epoch number
        """
        self._bound_functions['after_train'] = f
        return f

    def after_val(self, f):
        """
        Registers a hook called after every validation pass. Supported parameters:
        `metrics` - dict of this epoch's synchronized validation metrics, can be mutated
        `model` - the model
        `device` - device that the model/data resides on
        `epoch` - current epoch number
        """
        self._bound_functions['after_val'] = f
        return f
