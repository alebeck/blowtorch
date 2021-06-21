import warnings
from datetime import datetime
from typing import Optional, List, Union
from pathlib import Path
import random
import functools

import numpy as np
import torch
from torch.utils.data import DataLoader
from coolname import generate_slug, replace_random

from . import _writer as writer
from .backends.cpu_backend import CPUBackend
from .backends.gpu_backend import GPUBackend
from .utils import make_wrapper, get_highest_run, std_round, seed_all, set_deterministic
from .config import TrainingConfig
from .bound_functions import BoundFunctions
from .loggers import BaseLogger, LoggerSet, StandardLogger


class Run:
    """
    Represents an individual training run.
    """

    def __init__(self, config_files: Optional[List], random_seed: int = None):
        self._bound_functions = BoundFunctions()
        self._config = None
        self._backend = None
        self._model = None
        self._logger = None
        self._max_epochs = None
        self._use_gpu = None
        self._gpu_id = None
        self._resume_checkpoint = None
        self._save_path = None
        self._run_name = None
        self._optimize_metric = None
        self._checkpoint_metric = None
        self._checkpoint_every = None
        self._smaller_is_better = None  # TODO state which to minimize/checkpoint on in result dict
        self._optimize_first = None
        self._enable_amp = None
        self._is_validate = None
        self._config_files = []

        # TODO support run() args in config
        if config_files is None:
            config_files = []
        for file in config_files:
            self.add_config(file)
        self._config = TrainingConfig(self._config_files)

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
            train_loader: DataLoader,
            val_loader: DataLoader,
            loggers: Optional[List[BaseLogger]] = None,
            max_epochs=1,
            use_gpu=True,
            gpu_id=0,
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
            train_loader:
            val_loader:
            loggers: list of loggers that subscribe to various logging events
            max_epochs:
            use_gpu:
            gpu_id:
            resume_checkpoint: path to checkpoint to resume training from
            save_path: path to directory that blowtorch will save logs and checkpoints to
            run_name: name associated with this run, will be randomly created if None
            optimize_metric: train metric that will be used for optimization, will pick the first returned one if None
            checkpoint_metric: validation metric that will be used for checkpointing, will pick the first returned one if None
            checkpoint_every: every checkpoint_every epochs a checkpoint is saved, disregarding performance of the
            current model. This way it's always possible to resume the run from the latest (or near-latest) state
            smaller_is_better: ``True`` if we want to minimize, ``False`` if maximize
            optimize_first: whether optimization should occur during the first epoch
            enable_amp:
            detect_anomalies: enable autograd anomaly detection
        """
        self._model = model
        self._max_epochs = max_epochs
        self._use_gpu = use_gpu
        self._gpu_id = gpu_id
        self._resume_checkpoint = resume_checkpoint
        self._run_name = run_name
        self._save_path = save_path
        self._optimize_metric = optimize_metric
        self._checkpoint_metric = checkpoint_metric
        self._checkpoint_every = checkpoint_every
        self._smaller_is_better = smaller_is_better
        self._optimize_first = optimize_first
        self._enable_amp = enable_amp

        self._save_path = Path(self._save_path)
        self._save_path.mkdir(parents=True, exist_ok=True)

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

        checkpoints_path = self._save_path / 'checkpoints'
        checkpoints_path.mkdir(exist_ok=True)

        if loggers is None:
            loggers = []
        if not isinstance(loggers, (list, tuple)):
            loggers = [loggers]
        self._logger = LoggerSet([StandardLogger()] + loggers)
        self._logger.setup(self._save_path, self._run_name, self._resume_checkpoint is not None)

        if 'train_step' in self._bound_functions and 'val_step' in self._bound_functions:
            mode = 'step'
        elif 'train_epoch' in self._bound_functions and 'val_epoch' in self._bound_functions:
            mode = 'epoch'
        else:
            err_str = 'Need to register either both train_step/val_step or both train_epoch/val_epoch functions.'
            writer.error(err_str)
            raise ValueError(err_str)

        # Backend initialization
        try:
            if self._use_gpu:
                self._backend = GPUBackend(self._gpu_id, self._enable_amp)
            else:
                self._backend = CPUBackend()
        except Exception as e:
            writer.error(str(e))
            raise

        # load checkpoint if requested
        checkpoint = None
        start_epoch = 0
        if self._resume_checkpoint:
            writer.info(f'Resuming training from checkpoint {self._resume_checkpoint}')
            checkpoint = torch.load(checkpoints_path / 'latest', map_location='cpu')
            start_epoch = checkpoint['next_epoch']

        # backend takes care of initializing model, optimizers and schedulers
        self._backend.setup(self._model, self._bound_functions['configure_optimizers'], checkpoint)
        writer.info(f'Using {self._backend.get_name()}')

        if not self._optimize_first and start_epoch == 0:
            writer.info('Not optimizing during first epoch')

        self._logger.before_training_start(self._config.get_raw_config(), self._model, self._bound_functions)

        best_val = float('inf') if self._smaller_is_better else 0.

        for epoch in range(start_epoch, start_epoch + self._max_epochs):
            metrics = {}  # stores metrics of current epoch

            # train
            self._model.train()

            with writer.task(f'Training epoch {epoch}') as t:
                if mode == 'step':
                    step_metrics = []
                    for batch in t.tqdm(train_loader):
                        batch = self._backend.to_device(batch)

                        with torch.autograd.set_detect_anomaly(detect_anomalies):
                            # don't calculate grads if we're in epoch zero and not optimizing
                            torch.set_grad_enabled(self._optimize_first or epoch > 0)

                            train_metrics = self._backend.train_step(
                                self._bound_functions['train_step'],
                                batch=batch,
                                model=self._model,
                                is_validate=False,
                                device=self._backend.device,
                                epoch=epoch
                            )
                            assert isinstance(train_metrics, dict), '"train_step" should return a metrics dict.'

                            if self._optimize_metric is None:
                                metric = list(train_metrics.keys())[0]  # TODO possibility to state which one to optimize
                                writer.info(f'Selected metric "{metric}" for minimization')
                                self._optimize_metric = metric

                            if self._optimize_first or epoch > 0:
                                self._backend.optim_step(train_metrics[self._optimize_metric])

                        t.set_current_metrics({
                            self._optimize_metric: std_round(train_metrics[self._optimize_metric].item())})
                        step_metrics.append({k: float(v) for k, v in train_metrics.items()})

                        if 'after_train_step' in self._bound_functions:
                            self._bound_functions['after_train_step'](
                                model=self._model,
                                is_validate=False,
                                device=self._backend.device,
                                epoch=epoch
                            )

                    # calculate mean metrics
                    metrics['train'] = {
                        metric: np.array([dic[metric] for dic in step_metrics]).mean() for metric in step_metrics[0]
                    }

                elif mode == 'epoch':
                    # train_metrics = self._bound_functions['train_epoch'](data_loader=train_loader, model=self._model,
                    #                                                    is_validate=False, optimizers=self._optimizers)
                    # assert isinstance(train_metrics, dict), '"train_epoch" should return a metrics dict.'
                    # metrics['train'] = {k: float(v) for k, v in train_metrics.items()}
                    raise NotImplementedError()  # TODO wrap dataloader generator ?

                self._logger.after_pass(metrics['train'], epoch, is_validate=False)

                status_str = f'[Epoch {epoch} / Train] ' \
                             + ' '.join([f'{k}: {std_round(v)}' for k, v in metrics['train'].items()])
                t.success(status_str)

            # val
            self._model.eval()

            with writer.task(f'Validating epoch {epoch}') as t:
                if mode == 'step':
                    step_metrics = []
                    for batch in t.tqdm(val_loader):
                        batch = self._backend.to_device(batch)

                        val_metrics = self._backend.val_step(
                            self._bound_functions['val_step'],
                            batch=batch,
                            model=self._model,
                            is_validate=True,
                            device=self._backend.device,
                            epoch=epoch
                        )

                        assert isinstance(val_metrics, dict), '"val_step" should return a metrics dict.'
                        t.set_current_metrics({
                            self._optimize_metric: std_round(val_metrics[self._optimize_metric].item())})
                        step_metrics.append({k: float(v) for k, v in val_metrics.items()})

                    # calculate mean metrics
                    # todo agg function
                    metrics['val'] = {
                        metric: np.array([dic[metric] for dic in step_metrics]).mean() for metric in step_metrics[0]
                    }

                elif mode == 'epoch':
                    # val_metrics = self._bound_functions['val_epoch'](data_loader=val_loader, model=self._model,
                    #                                                  is_validate=True, optimizers=self._optimizers)
                    # assert isinstance(val_metrics, dict), '"val_epoch" should return a metrics dict.'
                    # metrics['val'] = {k: float(v) for k, v in val_metrics.items()}
                    raise NotImplementedError()

                # TODO specify metric to do scheduling on
                # if self._optimize_first is False, a warning will be raised by the schedulers which suggests that
                # optim.step() is called after scheduler.step(), which would normally result in the first epoch being
                # skipped from the learning rate scheduler. In our case, however, optim.step() was not called because
                # of self._optimize_first is False, and the epoch counter should indeed be increased.
                if epoch == 0 and not self._optimize_first:
                    warnings.simplefilter(action='ignore', category=UserWarning)
                    self._backend.scheduler_step(metrics['val'][self._optimize_metric])
                    warnings.filterwarnings('default')
                else:
                    self._backend.scheduler_step(metrics['val'][self._optimize_metric])

                self._logger.after_pass(metrics['val'], epoch, is_validate=True)

                status_str = f'[Epoch {epoch} / Val]   ' \
                             + ' '.join([f'{k}: {std_round(v)}' for k, v in metrics['val'].items()])
                t.success(status_str)

            if self._checkpoint_metric is None:
                metric = list(val_metrics.keys())[0]
                writer.info(f'Selected metric "{metric}" for checkpointing')
                self._checkpoint_metric = metric

            is_best = (self._smaller_is_better and metrics['val'][self._checkpoint_metric] < best_val) or \
                      (not self._smaller_is_better and metrics['val'][self._checkpoint_metric] > best_val)

            if is_best or epoch % self._checkpoint_every == 0:
                with writer.task(f'Saving checkpoint'):
                    checkpoint = {
                        'model': self._model.state_dict(),
                        'optimizers': {name: optim.state_dict() for name, optim in self._backend.optimizers.items()},
                        'schedulers': {name: sched.state_dict() for name, sched in self._backend.schedulers.items()},
                        'next_epoch': epoch + 1
                    }
                    path = checkpoints_path / f'epoch_{epoch}.pt'
                    torch.save(checkpoint, path)

                    latest_path = (checkpoints_path / 'latest')
                    best_path = (checkpoints_path / 'best')

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

                best_val = metrics['val'][self._checkpoint_metric]

        writer.success(f'Training finished')

    def add_config(self, path):
        self._config_files.append(path)

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

    def init(self, f):
        self._bound_functions['init'] = f
        return make_wrapper(f)

    def train_step(self, f):
        self._bound_functions['train_step'] = f
        return make_wrapper(f)

    def after_train_step(self, f):
        self._bound_functions['after_train_step'] = f
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
