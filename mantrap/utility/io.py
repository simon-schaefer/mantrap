import importlib
import inspect
import glob
import os
from typing import Callable, Dict


def build_os_path(filepath: str, make_dir: bool = False, free: bool = False) -> str:
    """Get path starting from home directory, i.e. get home directory and combine with given path.
    If the path does not exist, create it using os library.

    :param filepath: filepath starting from home directory.
    :param make_dir: create directory at output path (default = True).
    :param free: delete all files in directory if directory exists already.
    :return given path as absolute filepath.
    """
    home_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    path = os.path.join(home_directory, filepath)
    if make_dir:
        os.makedirs(path, exist_ok=True)
    if free:
        files = glob.glob(os.path.join(path, "*"))
        for f in files:
            os.remove(f)
    return path


def load_functions_from_module(module: str, prefix: str = None) -> Dict[str, Callable]:
    """Using importlib and inspect libraries load all functions (with prefix) from given module."""
    function_dict = {}
    module = importlib.import_module(module)
    functions = [o for o in inspect.getmembers(module) if inspect.isfunction(o[1])]
    for function_tuple in functions:
        function_name, _ = function_tuple
        if prefix is not None:
            if function_name.startswith(prefix):
                function_tag = function_name.replace(prefix, "")
                function_dict[function_tag] = function_tuple[1]
        else:
            function_dict[function_name] = function_tuple[1]
    return function_dict


def is_running_from_ipython():
    """Determine whether code running in jupyter notebook."""
    from IPython import get_ipython
    return get_ipython() is not None
