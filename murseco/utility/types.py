from typing import Any, Dict

from murseco.utility.io import JSONSerializer


class EnvActor(JSONSerializer):
    """Environment Actor object as a general type of any object in the environment.

    :argument category: obstacle, robot
    :argument element: object itself
    """

    def __init__(self, category: str, element: Any):
        super(EnvActor, self).__init__("utility/types/EnvActor")

        assert hasattr(element, "summary"), "object must have `summary` method"

        self.category = category
        self.element = element

    def summary(self) -> Dict[str, Any]:
        summary = super(EnvActor, self).summary()
        summary.update({"category": self.category, "element": self.element.summary()})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        super(EnvActor, cls).from_summary(json_text)
        return cls(json_text["category"], cls.call_by_summary(json_text["element"]))
