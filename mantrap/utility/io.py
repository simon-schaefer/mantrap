import glob
import os


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


def is_running_from_ipython():
    """Determine whether code running in jupyter notebook."""
    from IPython import get_ipython
    return get_ipython() is not None
