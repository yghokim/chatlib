import json
import re
from functools import cache
from typing import Type, Callable, Any, TypeVar

from pydantic import BaseModel, TypeAdapter

DataType = TypeVar('DataType')

BaseModelType = TypeVar('BaseModelType', bound=BaseModel)


@cache
def get_type_adapter(cls: Type[DataType]) -> TypeAdapter[DataType]:
    return TypeAdapter[cls]


def generate_pydantic_converter(cls: Type[BaseModelType]) -> tuple[
    Callable[[str, Any], BaseModelType], Callable[[BaseModelType, Any], str]]:
    return (lambda input, params: cls(**str_to_json_dict_converter(input, params))), (
        lambda input, params: input.json())


def generate_type_converter(cls: Type[DataType]) -> tuple[
    Callable[[str, Any], DataType], Callable[[DataType, Any], str]]:
    return (lambda input, params: get_type_adapter(cls).validate_json(input)), (
        lambda input, params: input.dump_json())


markdown_json_block_pattern = r'^```(json)?\s*(.*?)\s*```$'


def str_to_json_dict_converter(input: str, params: Any) -> dict:
    match = re.search(markdown_json_block_pattern, input, re.DOTALL)
    if match:
        return json.loads(match.group(2))
    else:
        return json.loads(input)


def json_dict_to_str_converter(input: dict, params: Any) -> str:
    return json.dumps(input, indent=2)


def str_to_str_noop(input: str, params: Any) -> str:
    return input

# class TestModel(BaseModel):
#    num: int
#    txt: str


# str_to_model, model_to_str = generate_pydantic_converter(TestModel)

# test_str = json.dumps({"txt": 123, "num": "hahaha", "hihi": "huhu"})

# print(str_to_model(test_str, None).json())
