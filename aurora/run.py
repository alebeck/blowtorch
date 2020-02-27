from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader

from .utils import make_wrapper, get_by_path, get_logger
from .config import TrainingConfig
from .context import TrainingContext
from .bound_functions import BoundFunctions


class Run:

    def __init__(self):
        self._bound_functions = BoundFunctions()
        self._config = None
        self._context = None
        self._primary_metric = None
        self._config_files = []

    def _dump_state(self):
        # TODO dump code, hw specs, ...
        config_path = Path(self._context['_log_path']) / 'config.yaml'
        with config_path.open('w') as fh:
            yaml.dump(self._config.get_raw_config(), fh)

    def _init_data(self):
        if isinstance(self._config['batch_size'], tuple):
            batch_size_train, batch_size_val = self._config['batch_size']
        else:
            batch_size_train, batch_size_val = (self._config['batch_size'],) * 2

        if 'train' not in self._config['data'] and 'val' not in self._config['data']:
            if not isinstance(self._config['val_size'], float):
                raise ValueError('"val_size" has to be specified when using automated train/val split.')

            ds = self._config['data']
            idx = np.arange(len(ds))
            np.random.shuffle(idx)
            split = int(self._config['val_size'] * len(ds))
            train_idx, val_idx = idx[split:], idx[:split]
            train_loader = DataLoader(ds, batch_size_train, sampler=SubsetRandomSampler(train_idx),
                                      num_workers=self._config['num_workers'], collate_fn=self._config['collate_fn'])
            val_loader = DataLoader(ds, batch_size_val, sampler=SubsetRandomSampler(val_idx),
                                    num_workers=self._config['num_workers'], collate_fn=self._config['collate_fn'])

        elif 'train' in self._config['data'] and 'val' in self._config['data']:
            ds_train = self._config['data']['train']
            ds_val = self._config['data']['val']
            train_loader = DataLoader(ds_train, batch_size_train, num_workers=self._config['num_workers'], shuffle=True,
                                      collate_fn=self._config['collate_fn'])
            val_loader = DataLoader(ds_val, batch_size_val, num_workers=self._config['num_workers'], shuffle=True,
                                    collate_fn=self._config['collate_fn'])

        else:
            raise ValueError('Received invalid dataset configuration.')

        return train_loader, val_loader

    def run(self):
        self._config = TrainingConfig(self._config_files)
        self._context = TrainingContext(self._config)

        counter = 1
        log_path = Path(self._config['log_path']) / datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        while log_path.exists():
            log_path = log_path / f'-{counter}'
            counter += 1

        log_path.mkdir(parents=True, exist_ok=True)
        self._context['_log_path'] = log_path

        logger = get_logger(log_path)

        self._dump_state()

        if 'train_step' in self._bound_functions and 'val_step' in self._bound_functions:
            mode = 'step'
        elif 'train_epoch' in self._bound_functions and 'val_epoch' in self._bound_functions:
            mode = 'epoch'
        else:
            err_str = 'Need to register either both train_step/val_step or both train_epoch/val_epoch functions.'
            logger.error(err_str)
            raise ValueError(err_str)

        start_epoch = 0

        # load checkpoint
        if self._config['resume']:
            with logger.task('Loading checkpoint'):
                checkpoint = torch.load(self._config['resume'])
                self._context['_model'].load_state_dict(checkpoint['model'])
                start_epoch = checkpoint['epoch']

        with logger.task('Instantiating dataset'):
            train_loader, val_loader = self._init_data()

        # check CUDA availability
        self._context['_use_cuda'] = self._config['use_cuda'] and torch.cuda.is_available()
        if self._context['_use_cuda']:
            self._context['_model'].cuda()
            logger.info('Using CUDA')

        if 'init' in self._bound_functions:
            self._bound_functions['init'](self._context)

        if self._config['resume']:
            logger.info(f'Resuming training from checkpoint {self._config["resume"]}')
            # resume optimizer state
            self._set_optim_states(checkpoint['optim'])
        else:
            logger.info('Starting training')

        self._primary_metric = self._config['primary_metric']
        best_val = float('inf') if self._config['smaller_is_better'] else 0.

        for epoch in range(start_epoch, start_epoch + self._config['epochs']):
            metrics = {}  # stores metrics of current epoch

            # train
            self._context['_model'].train()
            self._context['_is_validate'] = False

            with logger.task(f'Training epoch {epoch}') as t:
                if mode == 'auto':
                    raise NotImplementedError()

                elif mode == 'step':
                    step_metrics = []
                    for batch in t.tqdm(train_loader):
                        self._context['_batch'] = batch
                        train_metrics = self._bound_functions['train_step'](self._context)
                        assert isinstance(train_metrics, dict), '"train_step" should return a metrics dict.'
                        step_metrics.append({k: float(v) for k, v in train_metrics.items()})

                    # calculate mean metrics
                    metrics['train'] = {
                        metric: np.array([dic[metric] for dic in step_metrics]).mean() for metric in step_metrics[0]
                    }

                elif mode == 'epoch':
                    self._context['_data_loader'] = train_loader
                    train_metrics = self._bound_functions['train_epoch'](self._context)
                    assert isinstance(train_metrics, dict), '"train_epoch" should return a metrics dict.'
                    metrics['train'] = {k: float(v) for k, v in train_metrics.items()}

                # logging
                log_str = f'[Epoch {epoch} / Train] ' + \
                          ' '.join([f'{k}: {v}' for k, v in metrics['train'].items()])
                t.info(log_str)

            # val
            self._context['_model'].eval()
            self._context['_is_validate'] = True

            with logger.task(f'Validating epoch {epoch}') as t:
                if mode == 'auto':
                    raise NotImplementedError()

                elif mode == 'step':
                    step_metrics = []
                    for batch in t.tqdm(val_loader):
                        self._context['_batch'] = batch
                        val_metrics = self._bound_functions['val_step'](self._context)
                        assert isinstance(val_metrics, dict), '"val_step" should return a metrics dict.'
                        step_metrics.append({k: float(v) for k, v in val_metrics.items()})

                    # calculate mean metrics
                    metrics['val'] = {
                        metric: np.array([dic[metric] for dic in step_metrics]).mean() for metric in step_metrics[0]
                    }

                elif mode == 'epoch':
                    self._context['_data_loader'] = val_loader
                    val_metrics = self._bound_functions['val_epoch'](self._context)
                    assert isinstance(val_metrics, dict), '"val_epoch" should return a metrics dict.'
                    metrics['val'] = {k: float(v) for k, v in val_metrics.items()}

                # logging
                log_str = f'[Epoch {epoch} / Val] ' + \
                          ' '.join([f'{k}: {v}' for k, v in metrics['val'].items()])
                t.info(log_str)

            # dump metrics as yaml
            with (log_path / 'metrics.yaml').open('a') as fh:
                fh.write(yaml.dump({epoch: metrics}))

            if self._primary_metric is None:
                metric = list(val_metrics.keys())[0]
                logger.info(f'Using "{metric}" as primary metric')
                self._primary_metric = metric

            # save checkpoint
            if self._config['save_every'] or \
                    (self._config['smaller_is_better'] and val_metrics[self._primary_metric] < best_val) or \
                    (not self._config['smaller_is_better'] and val_metrics[self._primary_metric] > best_val):

                with logger.task(f'Saving checkpoint at epoch {epoch}'):
                    checkpoint = {
                        'model': self._context['_model'].state_dict(),
                        'optim': self._get_optim_states(),
                        'epoch': epoch + 1
                    }
                    checkpoint_path = log_path / 'checkpoints' / f'epoch_{epoch}.pt'
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, checkpoint_path)

                best_val = val_metrics[self._primary_metric]

        logger.success(f'Training finished')

    def __call__(self):
        self.run()

    def add_config(self, path):
        self._config_files.append(path)

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

    def _get_optim_states(self):
        optimizers = {}

        def get_recursive(dic, path=[]):
            for key, value in dic.items():
                if isinstance(value, dict):
                    get_recursive(value, path + [key])
                elif issubclass(type(value), torch.optim.Optimizer):
                    optimizers['.'.join(path + [key])] = value.state_dict()

        get_recursive(self._context)
        return optimizers

    def _set_optim_states(self, state_dicts):
        for path, state in state_dicts:
            optimizer = get_by_path(self._context, path)
            optimizer.load_state_dict(state)

