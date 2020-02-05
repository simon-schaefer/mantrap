import logging
import os


def add_coloring_to_ansi(fn):
    """Coloring ANSI text according to description to code in
    https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output.
    """
    def colored_string(*args):
        if type(args[1]) == logging.LogRecord:  # IPOPT weird callbacks output
            if type(args[1].msg) == bytes:
                return
        level = args[1].levelno
        if level >= 40:
            color = "\x1b[31m"  # red
        elif level >= 30:
            color = "\x1b[33m"  # yellow
        elif level >= 20:
            color = "\x1b[0m"  # normal
        else:
            color = "\x1b[38;5;247m"  # opaque
        args[1].msg = color + args[1].msg + "\x1b[0m"  # normal
        return fn(*args)
    return colored_string


def path_from_home_directory(filepath: str, make_dir: bool = True) -> str:
    """Get path starting from home directory, i.e. get home directory and combine with given path.
    If the path does not exist, create it using os library.

    :param filepath: filepath starting from home directory.
    :param make_dir: create directory at output path (default = True).
    :return given path as absolute filepath.
    """
    home_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    path = os.path.join(home_directory, filepath)
    if make_dir:
        os.makedirs(path, exist_ok=True)
    return path


def pytest_is_running() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ
