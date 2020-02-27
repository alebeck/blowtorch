from .config import TrainingConfig
from .bound_functions import in_bound_function
from .definitions import built_ins


class TrainingContext:

	_context = {}

	def __init__(self, config: TrainingConfig):
		for key, value in config.items():
			self._context[key] = value

		#for key in built_ins:
		#	self._context[key] = None

		self._context['_model'] = config['model']
		self._context['_context'] = self
		self._context['_'] = self

	def __getitem__(self, item):
		return self._context[item]

	def __setitem__(self, key, value):
		if in_bound_function() and key.startswith('_'):
			raise ValueError(f'Keys starting with "_" are reserved for built-in names. Tried to set "{key}".')
		self._context[key] = value

	# enable dot notation access
	def __getattr__(self, attr):
		return self[attr]

	def __setattr__(self, key, value):
		self[key] = value

	def keys(self):
		return self._context.keys()

	def items(self):
		return self._context.items()

	def get_dict(self):
		return self._context
