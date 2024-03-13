import json
from typing import Type, Callable, Any, TypeVar

from pydantic import BaseModel

BaseModelType = TypeVar('BaseModelType', bound=BaseModel)


def generate_pydantic_converter(cls: Type[BaseModelType]) -> tuple[Callable[[str, Any], BaseModelType], Callable[[BaseModelType, Any], str]]:
    return (lambda str, params: cls(**str_to_json_dict_converter(str))), (lambda input, params: input.json())


def str_to_json_dict_converter(input: str, params: Any) -> dict:
    return json.loads(input)


def json_dict_to_str_converter(input: dict, params: Any) -> str:
    return json.dumps(input, indent=2)

def str_to_str_noop(input: str, params: Any) -> str:
    return input

#class TestModel(BaseModel):
#    num: int
#    txt: str


#str_to_model, model_to_str = generate_pydantic_converter(TestModel)

#test_str = json.dumps({"txt": 123, "num": "hahaha", "hihi": "huhu"})

#print(str_to_model(test_str, None).json())
