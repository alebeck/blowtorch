from inspect import signature

from . import _writer as writer


AVAILABLE_PARAMS = {
    'train_step': ['batch', 'model', 'is_validate', 'device', 'epoch'],
    'after_train_step': ['model', 'is_validate', 'device', 'epoch'],
    'val_step': ['batch', 'model', 'is_validate', 'device', 'epoch'],
    'train_epoch': ['data_loader', 'model', 'is_validate', 'optimizers'],
    'val_epoch': ['data_loader', 'model', 'is_validate', 'optimizers'],
    'configure_optimizers': ['model']
}


def call(function, **kwargs):
    """
    Calls a function, only passing along the parameters matching the function's signature
    """
    kwargs = {k: v for k, v in kwargs.items() if k in signature(function).parameters}
    return function(**kwargs)


class BoundFunctions:

    def __init__(self):
        self.functions = {}

    def __setitem__(self, key, value):
        assert callable(value)
        # check for unavailable/undeclared context keys
        diff = set(signature(value).parameters).difference(AVAILABLE_PARAMS[key])
        if diff:
            err_str = f'Argument(s) {str(list(diff))[1:-1]} are not available for function ' \
                      f'\'{key}\'. Available parameters: {str(AVAILABLE_PARAMS[key])[1:-1]}.'
            writer.error(err_str)
            raise ValueError(err_str)

        self.functions[key] = value

    def __getitem__(self, item):
        return self.functions[item]

    def __contains__(self, item):
        return item in self.functions

    def items(self):
        return self.functions.items()

    def values(self):
        return self.functions.values()
