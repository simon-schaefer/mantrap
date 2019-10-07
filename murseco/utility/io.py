import os


def get_home_directory() -> str:
    """Get directory path of home directory i.e. the top-level of the project.
    :return home directory path as string.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def path_from_home_directory(filepath: str) -> str:
    """Get path starting from home directory, i.e. get home directory and combine with given path.
    :argument filepath: filepath starting from home directory.
    :return given path as absolute filepath.
    """
    return os.path.join(get_home_directory(), filepath)
