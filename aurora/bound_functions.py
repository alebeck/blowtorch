from inspect import signature

from .utils import get_logger


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
        Returns a callable which, when called, automatically injects the given context into the function.
        """
        def wrapper(context):
            global _in_bound_function

            params = self.signatures[item].parameters

            # # check for invalid built-ins
            # invalid = []
            # for param in params:
            #     if param.startswith('_') and param not in built_ins:
            #         invalid.append(param)
            # if invalid:
            #     err_str = f'Invalid built-in context keys found: {list(invalid)}. ' \
            #               f'Valid built-in keys are: {list(built_ins)}.'
            #     get_logger().error(err_str)
            #     raise ValueError(err_str)

            # check for unavailable/undeclared context keys
            missing = set(params).difference(context.keys())

            unavailable_builtins = {m for m in missing if m.startswith('_')}
            if unavailable_builtins:
                err_str = f'Built-in context keys {list(unavailable_builtins)} are not available for function ' \
                          f'\'{item}\'. Available keys: {[k for k in context.keys() if k.startswith("_")]}'
                get_logger().error(err_str)
                raise ValueError(err_str)

            undeclared_configs = missing.difference(unavailable_builtins)
            if undeclared_configs:
                err_str = f'Missing following keys in context for \'{item}\': {list(undeclared_configs)}. Have you ' \
                          f'declared them in your configuration or your init function?'
                get_logger().error(err_str)
                raise ValueError(err_str)

            kwargs = {k: v for k, v in context.items() if k in params}

            _in_bound_function = True
            return_val = self.functions[item](**kwargs)
            _in_bound_function = False

            return return_val
        return wrapper

    def __contains__(self, item):
        return item in self.functions
