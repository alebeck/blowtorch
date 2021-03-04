from inspect import signature

from .utils import get_terminal_writer


AVAILABLE_PARAMS = {
    'train_step': ['batch', 'model', 'is_validate', 'device', 'epoch'],
    'after_train_step': ['model', 'is_validate', 'device', 'epoch'],
    'val_step': ['batch', 'model', 'is_validate', 'device', 'epoch'],
    'train_epoch': ['data_loader', 'model', 'is_validate', 'optimizers'],
    'val_epoch': ['data_loader', 'model', 'is_validate', 'optimizers'],
    'configure_optimizers': ['model']
}
_in_bound_function = False


# will tell other modules whether we're currently inside a bound function.
def in_bound_function():
    return _in_bound_function


class BoundFunctions:

    def __init__(self):
        self.functions = {}
        self.signatures = {}

    def __setitem__(self, key, value):
        assert callable(value)
        self.signatures[key] = signature(value)
        self.functions[key] = value

    def __getitem__(self, item):
        """
        Returns a callable which, when called, automatically injects the given kwargs into the function.
        """
        def wrapper(**kwargs):
            global _in_bound_function

            used_params = self.signatures[item].parameters
            available_params = AVAILABLE_PARAMS[item]  # TODO catch item not in ALLOWED_PARAMS

            # check for unavailable/undeclared context keys
            missing = set(used_params).difference(available_params)
            if missing:
                err_str = f'Parameters {list(missing)} are not available for function ' \
                          f'\'{item}\'. Available parameters: {AVAILABLE_PARAMS[item]}'
                get_terminal_writer().error(err_str)
                raise ValueError(err_str)

            kwargs = {k: v for k, v in kwargs.items() if k in used_params}

            _in_bound_function = True
            return_val = self.functions[item](**kwargs)
            _in_bound_function = False

            return return_val
        return wrapper

    def __contains__(self, item):
        return item in self.functions

    def items(self):
        return self.functions.items()

    def values(self):
        return self.functions.values()

