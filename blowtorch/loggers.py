import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .bound_functions import BoundFunctions
from . import _writer as writer


# TODO implement standardlogger and always use it


class BaseLogger(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def setup(self, save_path: Path, run_name: str):
        """
        Used to setup the logger, given the run name
        """
        pass

    @abstractmethod
    def before_training_start(self, config: dict, model: nn.Module, bound_functions: BoundFunctions):
        pass

    @abstractmethod
    def after_pass(self, metrics: dict, epoch: int, is_validate: bool = False):
        pass

    @abstractmethod
    def after_epoch(self, epoch: int, model: nn.Module):
        pass


class StandardLogger(BaseLogger):

    def __init__(self):
        super().__init__()
        self._save_path = None
        self._log_file = None

    def setup(self, save_path: Path, run_name: str):
        self._save_path = save_path
        self._log_file = self._save_path / 'log.txt'

    def before_training_start(self, config: dict, model: nn.Module, bound_functions: BoundFunctions):
        # save config to file
        with (self._save_path / 'config.yaml').open('w') as fh:
            yaml.dump(config, fh)

        # save model summary and source to file
        with (self._save_path / 'model_summary.txt').open('w') as fh:
            fh.write(repr(model))

        with (self._save_path / 'source.txt').open('a') as fh:
            fh.write(inspect.getsource(type(model)))
            for function in bound_functions.values():
                fh.write('\n')
                fh.write(inspect.getsource(function))

    def after_pass(self, metrics: dict, epoch: int, is_validate: bool = False):
        status_str = f'[Epoch {epoch} / {"Val" if is_validate else "Train"}] ' \
                     + ' '.join([f'{k}: {v}' for k, v in metrics.items()]) + '\n'
        with self._log_file.open('a') as fh:
            fh.write(status_str)

    def after_epoch(self, epoch: int, model: nn.Module):
        pass


class WandbLogger(BaseLogger):

    def __init__(self, **kwargs):
        super().__init__()
        import wandb
        self.wandb = wandb
        self._wandb_args = kwargs

    def setup(self, save_path: Path, run_name: str):
        if 'name' in self._wandb_args:
            del self._wandb_args['name']
        self.wandb.init(name=run_name, dir=save_path, **self._wandb_args)

    def before_training_start(self, config: dict, model: nn.Module, bound_functions: BoundFunctions):
        self.wandb.config.update(config)
        # self.wandb.watch(model)

    def after_pass(self, metrics: dict, epoch: int, is_validate: bool = False):
        prefix = 'val_' if is_validate else 'train_'
        self.wandb.log({(prefix + k): v for k, v in metrics.items()}, step=epoch)

    def after_epoch(self, epoch: int, model: nn.Module):
        pass


class TensorBoardLogger(BaseLogger):

    def __init__(self, log_dir=None, **kwargs):
        """
        Initializes a new TensorBoard logger.
        :param log_dir: create a subfolder for each run inside this directory. Defaults to run.save_path.
        :param **kwargs: arguments directly passed on to SummaryWriter.
        """
        super().__init__()
        self.summary_writer = None
        self.log_dir = log_dir
        self._kwargs = kwargs

    def setup(self, save_path: Path, run_name: str):
        if self.log_dir is None:
            # use blowtorch save_path
            log_dir = save_path
        else:
            log_dir = Path(self.log_dir) / save_path.name

        writer.info(f'Using {log_dir} as TensorBoard log_dir')
        self.summary_writer = SummaryWriter(str(log_dir), **self._kwargs)

    def before_training_start(self, config: dict, model: nn.Module, bound_functions: BoundFunctions):
        pass

    def after_pass(self, metrics: dict, epoch: int, is_validate: bool = False):
        postfix = '/val' if is_validate else '/train'
        for k, v in metrics.items():
            self.summary_writer.add_scalar(k + postfix, v, epoch)

    def after_epoch(self, epoch: int, model: nn.Module):
        pass


class LoggerSet(BaseLogger):

    def __init__(self, loggers: List[BaseLogger]):
        super().__init__()
        self._loggers = loggers

    def setup(self, save_path: Path, run_name: str):
        for logger in self._loggers:
            logger.setup(save_path, run_name)

    def before_training_start(self, config: dict, model: nn.Module, bound_functions: BoundFunctions):
        for logger in self._loggers:
            logger.before_training_start(config, model, bound_functions)

    def after_pass(self, metrics: dict, epoch: int, is_validate: bool = False):
        for logger in self._loggers:
            logger.after_pass(metrics, epoch, is_validate)

    def after_epoch(self, epoch: int, model: nn.Module):
        for logger in self._loggers:
            logger.after_epoch(epoch, model)
