from .utils import get_terminal_writer

__version__ = '0.1'

_writer = get_terminal_writer()

from .run import Run
from .config import TrainingConfig
