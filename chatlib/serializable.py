from typing import Any
import json

class Serializable:
    def to_dict(self):
        return self.__dict__

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(recur_to_serializable_dict(self), indent=indent)

def recur_to_serializable_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: recur_to_serializable_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recur_to_serializable_dict(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(recur_to_serializable_dict(v) for v in obj)
    elif isinstance(obj, Serializable):
        return {k: recur_to_serializable_dict(v) for k, v in obj.to_dict().items()}
    else:
        return obj
