from abc import abstractmethod
import importlib
import os
from typing import Any, Dict

import json
import numpy as np

import murseco.utility.misc


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

    def __init__(self, name: str, is_unique: bool = True, identifier: str = None, color: str = None):
        self._name = name
        if is_unique:
            self._identifier = murseco.utility.misc.random_string() if identifier is None else identifier
            self._color = np.random.choice(murseco.utility.misc.MATPLOTLIB_COLORS) if color is None else color
        else:
            self._identifier = self._color = "none"

    @property
    def name(self) -> str:
        return self._name

    @property
    def color(self) -> str:
        return self._color

    @property
    def identifier(self) -> str:
        return self._identifier

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        """Summarize object in json-like dictionary file (should contain "name" key)."""
        return {"name": self._name, "identifier": self._identifier, "color": self._color}

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        """Load object parameters from json-style dictionary object."""
        return {"name": json_text["name"], "identifier": json_text["identifier"], "color": json_text["color"]}

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

    @staticmethod
    def call_by_summary(json_text: Dict[str, Any]):
        class_desc = str(json_text["name"]).replace("/", ".")
        class_lib, class_name = class_desc[: class_desc.rfind(".")], class_desc[class_desc.rfind(".") + 1 :]
        module = importlib.import_module("murseco." + class_lib)
        return getattr(getattr(module, class_name), "from_summary")(json_text)
