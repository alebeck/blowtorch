import ast
import copy
import sys

import yaml

from . import _writer
from .utils import deep_merge, get_by_path, set_by_path


class TrainingConfig:

    def __init__(self, config_files):
        self._config = {}

        # read in all config files
        for file in config_files:
            self._add_config(file)

        self._parse_cmd_args()
        self._raw_config = copy.deepcopy(self._config)

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

        overwritten = []

        # parse each config option individually
        for option in config_options:
            before, match, after = option.partition('=')
            if match == '=':
                try:
                    value = ast.literal_eval(after)
                except (ValueError, SyntaxError):
                    value = after  # interpret as string
                if before in self:
                    overwritten.append(before)
                set_by_path(self._config, before, value)
            else:
                # named config
                raise NotImplementedError

        if len(overwritten):
            _writer.info(f'Overwriting configuration{"s" if len(overwritten) > 1 else ""} {", ".join(overwritten)} '
                         f'from command line arguments')

    def __contains__(self, item):
        try:
            _ = self[item]
            return True
        except KeyError:
            return False

    def __getitem__(self, item):
        return get_by_path(self._config, item)

    def items(self):
        return self._config.items()

    def __repr__(self):
        return f'{__class__.__name__}(' + ', '.join([f'{k}={repr(v)}' for k, v in self._config.items()]) + ')'
