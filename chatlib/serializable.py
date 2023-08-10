from typing import Any


class Serializable:
    def to_dict(self):
        return self.__dict__


def recur_to_serializable_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: recur_to_serializable_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recur_to_serializable_dict(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(recur_to_serializable_dict(v) for v in obj)
    elif isinstance(obj, Serializable):
        return obj.to_dict()
    else:
        return obj
