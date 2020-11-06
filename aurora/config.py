from importlib import import_module
import ast
import copy
import sys

import yaml

from .utils import deep_merge, set_by_path


# These default values are automatically set if not specified explicitly.
_defaults = {'random_seed': 12345}

# These keys are required and have to be set within the configuration.
_required = []


class TrainingConfig:

    def __init__(self, config_files):
        self._config = {}

        # read in all config files
        for file in config_files:
            self._add_config(file)

        self._parse_cmd_args()
        self._set_defaults()
        self._raw_config = copy.deepcopy(self._config)
        self._instantiate_components()

        self._check()

    def get_raw_config(self):
        return self._raw_config

    def _add_config(self, path):
        with open(path) as fh:
            yaml_dict = yaml.load(fh.read(), Loader=yaml.FullLoader)
        self._config = deep_merge(self._config, yaml_dict)

    def _parse_cmd_args(self):
        try:
            i = sys.argv.index('with')
        except ValueError:
            # no config options passed
            return

        config_options = sys.argv[(i + 1):]
        if len(config_options) == 0:
            return

        # parse each config option individually
        for option in config_options:
            before, match, after = option.partition('=')
            if match == '=':
                try:
                    value = ast.literal_eval(after)
                except (ValueError, SyntaxError):
                    value = after  # interpret as string
                set_by_path(self._config, before, value)
            else:
                # named config
                raise NotImplementedError

    def _set_defaults(self):
        for key, value in _defaults.items():
            if key not in self._config:
                self._config[key] = value

    def _instantiate_components(self):
        def recursive_init(dic, key):
            value = dic[key]
            if isinstance(value, dict) and '__type__' in value:
                assert isinstance(value['__type__'], str), '__type__ has to be of type "str".'

                parts = value['__type__'].split('.')
                assert len(parts) > 1, 'Unrecognized type definition'

                module_path = '.'.join(parts[:-1])
                type_name = parts[-1]
                _type = getattr(import_module(module_path), type_name)

                # initialize
                del value['__type__']
                dic[key] = _type(**value)
            elif isinstance(value, dict):
                for key in value.keys():
                    recursive_init(value, key)

        for key in self._config:
            recursive_init(self._config, key)

    def _check(self):
        missing = set(_required).difference(self._config.keys())
        if missing:
            raise ValueError(f'Training configuration is incomplete. Missing {missing}')

    def __getitem__(self, item):
        return self._config[item]

    def items(self):
        return self._config.items()

    def __repr__(self):
        return f'{__class__.__name__}(' + ', '.join([f'{k}={repr(v)}' for k, v in self._config.items()]) + ')'
