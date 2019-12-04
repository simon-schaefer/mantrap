from datetime import datetime
import os


def get_home_directory() -> str:
    """Get directory path of home directory i.e. the top-level of the project.
    :return home directory path as string.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def path_from_home_directory(filepath: str) -> str:
    """Get path starting from home directory, i.e. get home directory and combine with given path.
    :param filepath: filepath starting from home directory.
    :return given path as absolute filepath.
    """
    return os.path.join(get_home_directory(), filepath)


def datetime_name() -> str:
    """Return formatted string encoding the date and time."""
    return datetime.now().strftime("%d_%m_%Y @ %H_%M%_S")


def add_coloring_to_ansi(fn):
    """Coloring ANSI text according to description to code in
    https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output.
    """
    def new(*args):
        level = args[1].levelno
        if level >= 40:
            color = "\x1b[31m"  # red
        elif level >= 30:
            color = "\x1b[33m"  # yellow
        elif level >= 20:
            color = "\x1b[32m"  # green
        else:
            color = "\x1b[38;5;247m"  # normal
        args[1].msg = color + args[1].msg + "\x1b[0m"  # normal
        return fn(*args)
    return new
