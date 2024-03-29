from typing import Callable


def make_non_empty_string_validator(msg: str) -> Callable:
    return lambda text: True if len(text.strip()) > 0 else msg
