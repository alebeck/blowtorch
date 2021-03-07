import sys
import time
import random
from functools import wraps
from pathlib import Path

from halo import Halo
from tqdm import tqdm
import torch
import numpy as np


AMP_AVAILABLE = hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast")
IS_TTY = sys.stdout.isatty()
NON_TTY_UPDATE_INTERVAL = 8000  # milliseconds


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


def get_by_path(dic, path):
	"""
	Gets a key (can be specified via dot notation `path`) in a nested `dic`
	"""
	parts = path.split('.')
	loc = dic
	for part in parts:
		if part not in loc:
			raise KeyError(path)
		loc = loc[part]
	return loc


def set_by_path(dic, path, value):
	"""
	Sets a key (can be specified via dot notation `path`) in a nested `dic` to `value`
	"""
	parts = path.split('.')
	loc = dic
	for part in parts[:-1]:
		if part not in loc:
			loc[part] = {}
		loc = loc[part]

	loc[parts[-1]] = value


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

	def __init__(self, text):
		self._text = text
		# if we're not in a tty, increase update interval to reduce log file sizes
		self._halo = Halo(text, interval=(-1 if IS_TTY else NON_TTY_UPDATE_INTERVAL))
		self._current_metrics = None

	def __enter__(self):
		self._halo.start()
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self._halo._stop_spinner.is_set():
			return

		if exc_type is None:
			self._halo.succeed()
		else:
			self._halo.fail()

		time.sleep(0.1)
		self._halo.stop()

	def tqdm(self, iterable):
		orig_text = self._halo.text

		def callback(f_dic):
			# format: Training epoch 2 (batch 1233/56990, 10:00 elapsed, 11:31 left)
			remaining = (f_dic['total'] - f_dic['n']) / f_dic['rate'] if f_dic['rate'] and f_dic['total'] else 0
			remaining_str = tqdm.format_interval(remaining) if f_dic['rate'] else '?'
			elapsed_str = tqdm.format_interval(f_dic["elapsed"])
			if self._current_metrics is not None:
				metrics_text = ''.join(f'{k}: {v}, ' for k, v in self._current_metrics.items())
			else:
				metrics_text = ''
			self._halo.text = \
				f'{orig_text} ({metrics_text}' \
				f'batch {f_dic["n"] + 1}/{f_dic["total"]}, {elapsed_str} elapsed, {remaining_str} left)'

		return _TQDM(iterable, callback=callback)

	def set_current_metrics(self, metric_dict):
		self._current_metrics = metric_dict

	def info(self, text=None):
		self._halo.info(text)

	def warning(self, text=None):
		if text is None:
			text = self._text
		self._halo.warn(text)

	def success(self, text=None):
		if text is None:
			text = self._text
		self._halo.succeed(text)

	def error(self, text=None):
		if text is None:
			text = self._text
		self._halo.fail(text)


class _TerminalWriter:

	@staticmethod
	def task(text):
		return _Spinner(text)

	@staticmethod
	def info(text):
		Halo(text).info()

	@staticmethod
	def warning(text):
		Halo(text).warn()

	@staticmethod
	def success(text):
		Halo(text).succeed()

	@staticmethod
	def error(text):
		Halo(text).fail()


_logger = None


def get_terminal_writer():
	global _logger

	if _logger is not None:
		return _logger

	_logger = _TerminalWriter()
	return _logger


def get_highest_run(save_path: Path):
	highest = 0
	for path in save_path.iterdir():
		num = path.name.split('-')[-1]
		try:
			if int(num) > highest:
				highest = int(num)
		except:
			pass

	return highest


def std_round(value):
	return round(value, 4)


def seed_all(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.set_deterministic(True)
	try:
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
	except:
		pass
