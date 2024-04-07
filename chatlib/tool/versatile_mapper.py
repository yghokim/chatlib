import json
from itertools import chain
from typing import TypeVar, Generic, Callable, Any

from pydantic import BaseModel, ConfigDict

from chatlib.chatbot import ChatCompletionParams
from chatlib.llm.chat_completion_api import ChatCompletionAPI, ChatCompletionMessage, ChatCompletionMessageRole, \
    ChatCompletionFinishReason
from chatlib.tool.converter import str_to_str_noop


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
                 str_output_converter: Callable[[str, ParamsType], OutputType]
                 ):
        self.__api = api
        self.__instruction_generator = instruction_generator
        self.__str_output_converter = str_output_converter

        self.__input_str_converter = input_str_converter or str_to_str_noop
        self.__output_str_converter = output_str_converter

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
            example_messages = list(chain.from_iterable([[
                ChatCompletionMessage(content=self.__input_str_converter(example.input, params),
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
                    return self.__str_output_converter(chat_response.message.content, params)
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
