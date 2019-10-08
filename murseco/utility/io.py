from abc import abstractmethod
import os
from typing import Any, Dict

import json


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


class JSONSerializer:
    """The JSONSerializer class is an abstract class that gives an interface for dumping and loading objects
    to a json file, for storing parameters, environments, etc."""

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        """Summarize object in json-like dictionary file (should contain "name" key)."""
        pass

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        """Load object parameters from json-style dictionary object."""
        pass

    def to_json(self, filepath: str):
        """Write summary of given object to json file, containing the distribution type, parameters, etc.
        :argument filepath to write to.
        """
        file_object = open(filepath, "w")
        return json.dump(self.summary(), file_object)

    @classmethod
    def from_json(cls, filepath: str):
        """Load object from json file by reading it, transform it to a dict-object and load the parameters from there.
        :argument filepath to read from.
        """
        with open(filepath, "r") as read_file:
            json_text = json.load(read_file)
        return cls.from_summary(json_text)
