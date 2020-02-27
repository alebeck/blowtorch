import logging
import time
from functools import wraps
import traceback

from halo import Halo
from tqdm import tqdm


def make_wrapper(f):
	@wraps(f)
	def wrapper(*args, **kwargs):
		return f(*args, **kwargs)
	return wrapper


def deep_merge(a, b):
	"""
	Deep-merges two dicts, with b overwriting values in a.
	https://stackoverflow.com/a/20666342
	"""
	for key, value in b.items():
		if isinstance(value, dict):
			node = a.setdefault(key, {})
			deep_merge(node, value)
		else:
			a[key] = value
	return a


def set_by_path(dic, path, value):
	parts = path.split('.')
	loc = dic
	for part in parts[:-1]:
		if part not in loc:
			loc[part] = {}
		loc = loc[part]

	loc[parts[-1]] = value


def get_by_path(dic, path):
	parts = path.split('.')
	loc = dic
	for part in parts[:-1]:
		assert part in loc, f'Key {part} not found in dict.'
		loc = loc[part]

	return loc[parts[-1]]


class _LevelFormatter(logging.Formatter):

	def format(self, record):
		return f'[{record.levelname}] {record.msg}'


class _TQDM(tqdm):
	"""
	Provides a subclass of tqdm which returns current format dict to a callback instead of printing it.
	"""

	def __init__(self, iterable, callback=None):
		self.callback = callback
		super().__init__(iterable)

	def display(self, msg=None, pos=None):
		self.callback(self.format_dict)

	def close(self):
		pass


class _Spinner:

	def __init__(self, text, logger):
		self._text = text
		self._halo = Halo(text)
		self._logger = logger

	def __enter__(self):
		self._halo.start()
		self._logger.info(self._text)
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self._halo._stop_spinner.is_set():
			return

		if exc_type is None:
			self._halo.succeed()
			self._logger.info(self._text + ' - success')
		else:
			self._halo.fail()
			self._logger.error(self._text + ' - failed\n' + traceback.format_exc())

		time.sleep(0.1)
		self._halo.stop()

	def tqdm(self, iterable):
		orig_text = self._halo.text

		def callback(f_dic):
			# format: Training epoch 2 (batch 1233/56990, 11:31 left)
			remaining = (f_dic['total'] - f_dic['n']) / f_dic['rate'] if f_dic['rate'] and f_dic['total'] else 0
			remaining_str = tqdm.format_interval(remaining) if f_dic['rate'] else '?'
			self._halo.text = f'{orig_text} (batch {f_dic["n"] + 1}/{f_dic["total"]}, {remaining_str} left)'

		return _TQDM(iterable, callback=callback)

	def info(self, text=None):
		if text is not None:
			self._logger.info(text)
		self._halo.info(text)

	def warning(self, text=None):
		if text is None:
			text = self._text
		self._logger.warning(text + (' - warning' if text == self._text else ''))
		self._halo.warn(text)

	def success(self, text=None):
		if text is None:
			text = self._text
		self._logger.info(text + (' - success' if text == self._text else ''))
		self._halo.succeed(text)

	def error(self, text=None):
		if text is None:
			text = self._text
		self._logger.error(text + (' - failed' if text == self._text else ''))
		self._halo.fail(text)


class _Logger:

	def __init__(self, log_path):
		file_handler = logging.FileHandler(log_path / 'log.txt')
		file_handler.setFormatter(_LevelFormatter())

		self._logger = logging.getLogger('train_logger')
		self._logger.handlers = []
		self._logger.propagate = False
		self._logger.setLevel(logging.INFO)
		self._logger.addHandler(file_handler)

	def task(self, text):
		return _Spinner(text, self._logger)

	def info(self, text):
		self._logger.info(text)
		Halo(text).info()

	def warning(self, text):
		self._logger.warning(text)
		Halo(text).warn()

	def success(self, text):
		self._logger.info(text)
		Halo(text).succeed()

	def error(self, text):
		self._logger.error(text)
		Halo(text).fail()


_logger = None


def get_logger(log_path=None):
	global _logger

	if _logger is not None:
		return _logger

	assert log_path is not None
	_logger = _Logger(log_path)
	return _logger
