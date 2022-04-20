from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import time

from ruamel.yaml import YAML
from torch import nn

from .bound_functions import BoundFunctions
from . import _writer as writer

yaml = YAML(typ='safe')


class BaseLogger(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def setup(self, save_path: Path, run_name: str, resume: bool):
        """
        Used to setup the logger, given the run name
        """
        pass

    @abstractmethod
    def before_training_start(self, config: dict, model: nn.Module, bound_functions: BoundFunctions):
        pass

    @abstractmethod
    def after_pass(self, metrics: dict, charts: dict, epoch: int, is_validate: bool = False):
        pass

    @abstractmethod
    def after_epoch(self, epoch: int, model: nn.Module):
        pass


class StandardLogger(BaseLogger):

    def __init__(self):
        super().__init__()
        self._save_path = None
        self._log_file = None
        self._resume = None

    def setup(self, save_path: Path, run_name: str, resume: bool):
        self._save_path = save_path
        self._log_file = self._save_path / 'log.txt'
        self._resume = resume

    def before_training_start(self, config: dict, model: nn.Module, bound_functions: BoundFunctions):
        if self._resume:
            return

        # save config to file
        with (self._save_path / 'config.yaml').open('w') as fh:
            yaml.dump(config, fh)

        # save model summary and source to file
        with (self._save_path / 'model_summary.txt').open('w') as fh:
            fh.write(repr(model))

    def after_pass(self, metrics: dict, charts: dict, epoch: int, is_validate: bool = False):
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
        self._wandb = wandb
        self._wandb_args = kwargs
        self._resume = None

    def setup(self, save_path: Path, run_name: str, resume: bool, retries=5):
        if 'name' in self._wandb_args:
            del self._wandb_args['name']
        self._resume = resume
        # always set resume=True, since every run is saved in its own directory anyway and wandb will just create a new
        # run if no previous run exists in the save_path directory. This way wandb automatically saves the run id in
        # the save_path directory and automatically resumes later, if desired.
        for k in range(retries):
            try:
                self._wandb.init(name=run_name, dir=save_path, resume=True, **self._wandb_args)
                return
            except:
                time.sleep(30)

        raise ConnectionError(f'Cannot connect to wandb after {retries} retries')

    def before_training_start(self, config: dict, model: nn.Module, bound_functions: BoundFunctions):
        if not self._resume:
            self._wandb.config.update(config)

    def after_pass(self, metrics: dict, charts: dict, epoch: int, is_validate: bool = False):
        postfix = '/val' if is_validate else '/train'

        if charts is not None:
            self._wandb.log({k: v for k, v in charts.items()}, commit=False)

        self._wandb.log({(k + postfix): v for k, v in metrics.items()}, step=epoch)

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
        # do this to fail early in case tensorboard is not installed
        from torch.utils.tensorboard import SummaryWriter
        self.summary_writer = None
        self.log_dir = log_dir
        self._kwargs = kwargs

    def setup(self, save_path: Path, run_name: str, resume: bool):
        if self.log_dir is None:
            # use blowtorch save_path
            log_dir = save_path
        else:
            log_dir = Path(self.log_dir) / save_path.name

        writer.info(f'Using {log_dir} as TensorBoard log_dir')
        from torch.utils.tensorboard import SummaryWriter
        self.summary_writer = SummaryWriter(str(log_dir), **self._kwargs)

    def before_training_start(self, config: dict, model: nn.Module, bound_functions: BoundFunctions):
        pass

    def after_pass(self, metrics: dict, charts: dict, epoch: int, is_validate: bool = False):
        postfix = '/val' if is_validate else '/train'
        for k, v in metrics.items():
            self.summary_writer.add_scalar(k + postfix, v, epoch)

    def after_epoch(self, epoch: int, model: nn.Module):
        pass


class LoggerSet(BaseLogger):

    def __init__(self, loggers: List[BaseLogger]):
        super().__init__()
        self._loggers = loggers

    def setup(self, save_path: Path, run_name: str, resume: bool):
        for logger in self._loggers:
            logger.setup(save_path, run_name, resume)

    def before_training_start(self, config: dict, model: nn.Module, bound_functions: BoundFunctions):
        for logger in self._loggers:
            logger.before_training_start(config, model, bound_functions)

    def after_pass(self, metrics: dict, charts: dict, epoch: int, is_validate: bool = False):
        for logger in self._loggers:
            logger.after_pass(metrics, charts, epoch, is_validate)

    def after_epoch(self, epoch: int, model: nn.Module):
        for logger in self._loggers:
            logger.after_epoch(epoch, model)
