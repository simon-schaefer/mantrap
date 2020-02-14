import glob
import logging
import os


def build_output_path(filepath: str, make_dir: bool = True, free: bool = False) -> str:
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


def no_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" not in os.environ


def remove_bytes_from_logging(fn):
    """Remove weird IPOPT callbacks logging output (byte strings) from log."""
    def remove_bytes(*args):
        if type(args[1]) == logging.LogRecord and type(args[1].msg) == bytes:  #
            return
        return fn(*args)
    return remove_bytes
