import importlib
import inspect
import typing


def __load_from_module(module: str, prefix: str = None) -> typing.Dict[str, typing.Callable]:
    """Using importlib and inspect libraries load all functions (with prefix) from given module."""
    function_dict = {}
    module = importlib.import_module(module)
    functions = [o for o in inspect.getmembers(module)]
    for function_tuple in functions:
        function_name, _ = function_tuple
        if prefix is not None:
            if function_name.startswith(prefix):
                function_tag = function_name.replace(prefix, "")
                function_dict[function_tag] = function_tuple[1]
        else:
            function_dict[function_name] = function_tuple[1]
    return function_dict


def __dict_to_tuples(x: typing.Dict[str, typing.Any]) -> typing.List[typing.Tuple[str, typing.Any]]:
    return list(zip(x.keys(), x.values()))


def load_functions_from_module(module: str, prefix: str = None, as_tuples: bool = False
                               ) -> typing.Union[typing.List, typing.Dict[str, typing.Callable]]:
    object_dict = __load_from_module(module=module, prefix=prefix)
    object_dict = {name: c_object for name, c_object in object_dict.items() if inspect.isfunction(c_object)}
    return object_dict if not as_tuples else __dict_to_tuples(object_dict)


def load_class_from_module(module: str, prefix: str = None, as_tuples: bool = False
                           ) -> typing.Union[typing.List, typing.Dict[str, typing.Callable]]:
    object_dict = __load_from_module(module=module, prefix=prefix)
    object_dict = {name: c_object for name, c_object in object_dict.items() if inspect.isclass(c_object)}
    return object_dict if not as_tuples else __dict_to_tuples(object_dict)

