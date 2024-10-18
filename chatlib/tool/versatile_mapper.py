import json
from itertools import chain
from typing import TypeVar, Generic, Callable, Any

from pydantic import BaseModel, ConfigDict

from chatlib.chatbot import ChatCompletionParams, Dialogue, DialogueTurn
from chatlib.llm.chat_completion_api import ChatCompletionAPI, ChatCompletionMessage, ChatCompletionMessageRole, \
    ChatCompletionFinishReason
from chatlib.tool.converter import str_to_str_noop
from chatlib.utils.jinja_utils import convert_to_jinja_template


class ChatCompletionFewShotMapperParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    api_params: ChatCompletionParams


InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')
ParamsType = TypeVar('ParamsType', bound=ChatCompletionFewShotMapperParams)


class MapperInputOutputPair(BaseModel, Generic[InputType, OutputType]):
    model_config = ConfigDict(frozen=True)

    input: InputType
    output: OutputType


class ChatCompletionFewShotMapper(Generic[InputType, OutputType, ParamsType]):

    @classmethod
    def make_str_mapper(cls, api: ChatCompletionAPI,
                 instruction_generator: Callable[[str, ParamsType | None], str] | str) -> 'ChatCompletionFewShotMapper[str, str, ParamsType]':
        return ChatCompletionFewShotMapper(api, instruction_generator, str_to_str_noop, str_to_str_noop, str_to_str_noop)

    def __init__(self,
                 api: ChatCompletionAPI,
                 instruction_generator: Callable[[InputType, ParamsType | None], str] | str,
                 input_str_converter: Callable[[InputType, ParamsType], str] | None,
                 output_str_converter: Callable[[OutputType, ParamsType], str],
                 str_output_converter: Callable[[str, ParamsType], OutputType],
                 output_validator: Callable[[InputType, OutputType], bool] | None = None,
                 example_str_converter: Callable[[InputType, ParamsType], str] | None = None,
                 ):
        self.__api = api
        self.__instruction_generator = instruction_generator
        self.__str_output_converter = str_output_converter

        self.__input_str_converter = input_str_converter or str_to_str_noop
        self.__output_str_converter = output_str_converter

        self.__example_str_converter = example_str_converter

        self.__output_validator = output_validator

    @property
    def api(self) -> ChatCompletionAPI:
        return self.__api

    async def run(self,
                  examples: list[MapperInputOutputPair[InputType, OutputType]] | None,
                  input: InputType,
                  params: ParamsType,
                  output_malformed_retry_count: int = 5
                  ) -> OutputType:
        if examples is not None:  # TODO cache example messages
            converter = self.__example_str_converter or self.__input_str_converter
            example_messages = list(chain.from_iterable([[
                ChatCompletionMessage(content=converter(example.input, params),
                                      role=ChatCompletionMessageRole.SYSTEM, name="example_user"),
                ChatCompletionMessage(content=self.__output_str_converter(example.output, params),
                                      role=ChatCompletionMessageRole.SYSTEM, name="example_assistant")
            ] for example in examples]))
        else:
            example_messages = None

        if isinstance(self.__instruction_generator, str):
            instruction = self.__instruction_generator
        else:
            instruction = self.__instruction_generator(input, params)

        messages = [ChatCompletionMessage(content=instruction, role=ChatCompletionMessageRole.SYSTEM)]

        if example_messages is not None:
            messages.extend(example_messages)

        messages.append(ChatCompletionMessage(content=self.__input_str_converter(input, params),
                                              role=ChatCompletionMessageRole.USER))

        left_retry_count = output_malformed_retry_count
        while True:
            chat_response = await self.__api.run_chat_completion(params.model, messages, params.api_params.dict())

            if chat_response.finish_reason == ChatCompletionFinishReason.Stop:
                try:
                    output = self.__str_output_converter(chat_response.message.content, params)
                    if self.__output_validator is not None:
                        if self.__output_validator(input, output) is True:
                            return output
                        else:
                            raise ValueError("Output validation failed.")
                    else:
                        return output
                except Exception as e:  # If converting fails
                    if left_retry_count > 0:
                        print(
                            f"Output converting failed. retry count left: {left_retry_count}, Content: \"{chat_response.message.content}\"")
                        print(f"Error: {e}")
                        left_retry_count -= 1
                        continue
                    else:
                        raise Exception(
                            "Output malformed for conversion. Consumed all retry count. PLease check your instruction.")
            else:
                raise Exception(chat_response.finish_reason)


DEFAULT_USER_ALIAS = "User"
DEFAULT_SYSTEM_ALIAS = "AI"

DIALOGUE_TEMPLATE = convert_to_jinja_template("""
<dialogue>
{% for turn in dialogue %}
{%- if turn.is_user == true %}{{user_alias}}{%-else-%}{{system_alias}}{%-endif-%}: <msg>{{turn.message}}</msg>
{%endfor%}</dialogue>
""")

class DialogueSummarizer(ChatCompletionFewShotMapper[Dialogue, OutputType, ParamsType]):

    def __init__(self, 
                 api: ChatCompletionAPI,
                 instruction_generator: Callable[[Dialogue, ParamsType | None], str] | str,
                 output_str_converter: Callable[[OutputType, ParamsType], str],
                 str_output_converter: Callable[[str, ParamsType], OutputType],
                 output_validator: Callable[[InputType, OutputType], bool] | None = None,
                 dialogue_filter: Callable[[Dialogue, ParamsType | None], Dialogue] | None = None,
                 user_alias: str | None = None,
                 system_alias: str | None = None
                 ):

        self.__dialogue_filter = dialogue_filter
        self.__user_alias = user_alias
        self.__system_alias = system_alias

        user_alias = (
            self.__user_alias if self.__user_alias is not None and len(self.__user_alias) > 0 else DEFAULT_USER_ALIAS)
        system_alias = (
            self.__system_alias if self.__system_alias is not None and len(self.__system_alias) > 0 else DEFAULT_SYSTEM_ALIAS)
        

        super().__init__(api, instruction_generator, 
                         lambda d, p: DIALOGUE_TEMPLATE.render(user_alias=user_alias, system_alias=system_alias, dialogue=self.__dialogue_filter(d, p) if self.__dialogue_filter is not None else d), 
                         output_str_converter, str_output_converter, output_validator, 
                         lambda d, p: DIALOGUE_TEMPLATE.render(user_alias=user_alias, system_alias=system_alias, dialogue=d)
                         )