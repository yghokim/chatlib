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
            lambda input, params: input.json(
                **serialization_kwargs) if serialization_kwargs is not None else input.json())
    elif serialization_type == 'yaml':
        return (lambda input, params: cls(**yaml_str_to_dict_converter(input, params))), (
            lambda input, params: dict_to_yaml_str_converter(
                input.json(**serialization_kwargs) if serialization_kwargs is not None else input.json(), params))


def generate_pydantic_list_converter(cls_list: Type[list[BaseModelType]], cls_elm: Type[BaseModelType],
                                     serialization_type: Literal['json'] | Literal['yaml'] = 'json',
                                     serialization_kwargs: dict | None = None) -> tuple[
    Callable[[str, Any], list[BaseModelType]], Callable[[list[BaseModelType], Any], str]]:
    str_to_obj, obj_to_str = generate_type_converter(cls_list, serialization_type)

    if serialization_kwargs is None:
        return str_to_obj, obj_to_str
    else:
        def serialize(l: cls_list, params: Any) -> str:
            dict_list = [elm.model_dump(**serialization_kwargs) for elm in l]
            if serialization_type == 'json':
                return dict_to_json_str_converter(dict_list, params)
            elif serialization_type == 'yaml':
                return dict_to_yaml_str_converter(dict_list, params)

        return str_to_obj, serialize


def generate_type_converter(cls: Type[DataType], serialization_type: Literal['json'] | Literal['yaml'] = 'json') -> \
tuple[
    Callable[[str, Any], DataType], Callable[[DataType, Any], str]]:
    if serialization_type == 'json':
        return (lambda input, params: get_type_adapter(cls).validate_json(input)), (
            lambda input, params: get_type_adapter(cls).dump_json(input).decode('utf-8'))
    elif serialization_type == 'yaml':
        return (
            lambda input, params: get_type_adapter(cls).validate_python(yaml_str_to_dict_converter(input, params))), (
            lambda input, params: dict_to_yaml_str_converter(get_type_adapter(cls).dump_python(input), params))


markdown_json_block_pattern = r'^```(json)?\s*(.*?)\s*```$'
markdown_yaml_block_pattern = r'^```(yaml)?\s*(.*?)\s*```$'


def json_str_to_dict_converter(input: str, params: Any) -> dict:
    match = re.search(markdown_json_block_pattern, input, re.DOTALL)
    if match:
        return json.loads(match.group(2))
    else:
        return json.loads(input)


def dict_to_json_str_converter(input: dict | list, params: Any) -> str:
    return json.dumps(input, indent=2)


def yaml_str_to_dict_converter(input: str, params: Any) -> dict:
    match = re.search(markdown_yaml_block_pattern, input, re.DOTALL)
    if match:
        input = match.group(2)

    return yaml.load(io.StringIO(input), Loader=yaml.FullLoader)


def dict_to_yaml_str_converter(input: dict | list, params: Any) -> str:
    return yaml.dump(input, indent=2)


def str_to_str_noop(input: str, params: Any) -> str:
    return input

#class TestModel(BaseModel):
#   num: int
#   txt: str
#   other: str | None = None



#str_to_model, model_to_str = generate_pydantic_list_converter(list[TestModel], TestModel, serialization_kwargs=dict(include={"num", "txt"}))

#test_str = json.dumps([{"txt": '123', "num": 5}])

#print(str_to_model(test_str, None))
#print(model_to_str([TestModel(num=10, txt="hihi", other="hello")], None))
