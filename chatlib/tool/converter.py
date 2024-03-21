import io
import json
import re
from functools import cache
from typing import Type, Callable, Any, TypeVar, Literal

import yaml
from pydantic import BaseModel, TypeAdapter

DataType = TypeVar('DataType')

BaseModelType = TypeVar('BaseModelType', bound=BaseModel)


@cache
def get_type_adapter(cls: Type[DataType]) -> TypeAdapter[DataType]:
    return TypeAdapter(cls)


def generate_pydantic_converter(cls: Type[BaseModelType],
                                serialization_type: Literal['json'] | Literal['yaml'] = 'json',
                                serialization_kwargs: dict | None = None) -> tuple[
    Callable[[str, Any], BaseModelType], Callable[[BaseModelType, Any], str]]:

    if serialization_type == 'json':
        return (lambda input, params: cls(**json_str_to_dict_converter(input, params))), (
            lambda input, params: input.json(**serialization_kwargs))
    elif serialization_type == 'yaml':
        return (lambda input, params: cls(**yaml_str_to_dict_converter(input, params))), (
            lambda input, params: dict_to_yaml_str_converter(input.json(**serialization_kwargs), params))


def generate_type_converter(cls: Type[DataType], serialization_type: Literal['json'] | Literal['yaml'] = 'json') -> tuple[
    Callable[[str, Any], DataType], Callable[[DataType, Any], str]]:

    if serialization_type == 'json':
        return (lambda input, params: get_type_adapter(cls).validate_json(input)), (
            lambda input, params: get_type_adapter(cls).dump_json(input).decode('utf-8'))
    elif serialization_type == 'yaml':
        return (lambda input, params: get_type_adapter(cls).validate_python(yaml_str_to_dict_converter(input, params))), (
            lambda input, params: dict_to_yaml_str_converter(get_type_adapter(cls).dump_python(input), params))


markdown_json_block_pattern = r'^```(json)?\s*(.*?)\s*```$'
markdown_yaml_block_pattern = r'^```(yaml)?\s*(.*?)\s*```$'


def json_str_to_dict_converter(input: str, params: Any) -> dict:
    match = re.search(markdown_json_block_pattern, input, re.DOTALL)
    if match:
        return json.loads(match.group(2))
    else:
        return json.loads(input)


def dict_to_json_str_converter(input: dict, params: Any) -> str:
    return json.dumps(input, indent=2)


def yaml_str_to_dict_converter(input: str, params: Any) -> dict:
    match = re.search(markdown_yaml_block_pattern, input, re.DOTALL)
    if match:
        input = match.group(2)

    return yaml.load(io.StringIO(input))


def dict_to_yaml_str_converter(input: dict, params: Any) -> str:
    return yaml.dump(input, indent=2)


def str_to_str_noop(input: str, params: Any) -> str:
    return input

# class TestModel(BaseModel):
#    num: int
#    txt: str


# str_to_model, model_to_str = generate_pydantic_converter(TestModel)

# test_str = json.dumps({"txt": 123, "num": "hahaha", "hihi": "huhu"})

# print(str_to_model(test_str, None).json())
