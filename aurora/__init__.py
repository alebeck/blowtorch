from .utils import get_terminal_writer

__version__ = '0.0.2'

_writer = get_terminal_writer()

from .run import Run
from .config import TrainingConfig
