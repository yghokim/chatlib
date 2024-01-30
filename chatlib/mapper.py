import json
import string
from abc import ABC, abstractmethod
from itertools import chain
from json import JSONDecodeError
from typing import TypeVar, Generic, Callable

from chatlib.chat_completion import ChatCompletionMessage, ChatCompletionMessageRole
from chatlib.jinja_utils import convert_to_jinja_template
from jinja2 import Template

from chatlib.chatbot.generators import ChatGPTResponseGenerator

from chatlib.chatbot import DialogueTurn, Dialogue, RegenerateRequestException, TokenLimitExceedHandler
from chatlib.openai_utils import ChatGPTModel, ChatGPTParams

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')
ParamsType = TypeVar('ParamsType')


class Mapper(Generic[InputType, OutputType, ParamsType], ABC):

    @abstractmethod
    async def run(self, input: InputType, params: ParamsType | None = None) -> OutputType:
        pass


class ChatGPTFewShotLearnerParams:

    def __init__(self,
                 instruction_params: dict | None = None
                 ):
        self.instruction_params = instruction_params


class ChatGPTDialogSummarizerParams(ChatGPTFewShotLearnerParams):

    def __init__(self,
                 input_user_alias: str | None = None,
                 input_system_alias: str | None = None,
                 instruction_params: dict | None = None):
        super().__init__(instruction_params)
        self.input_user_alias = input_user_alias
        self.input_system_alias = input_system_alias


DEFAULT_USER_ALIAS = "User"
DEFAULT_SYSTEM_ALIAS = "AI"

DIALOGUE_TEMPLATE = convert_to_jinja_template("""
<dialogue>
{% for turn in dialogue %}
{%- if turn.is_user == true %}{{user_alias}}{%-else-%}{{system_alias}}{%-endif-%}: <msg>{{turn.message}}</msg>
{%endfor%}</dialogue>
""")

ChatGPTFewShotParamsType = TypeVar("ChatGPTFewShotParamsType", bound=ChatGPTFewShotLearnerParams)


# Map input to string using ChatGPT.
class ChatGPTFewShotMapper(Mapper[InputType, OutputType, ChatGPTFewShotParamsType],
                           Generic[InputType, OutputType, ChatGPTFewShotParamsType], ABC):

    def __init__(self,
                 base_instruction: str | Template,
                 model: str = ChatGPTModel.GPT_4_latest,
                 gpt_params: ChatGPTParams | None = None,
                 examples: list[tuple[InputType, str]] | None = None,
                 token_limit_exceed_handler: TokenLimitExceedHandler | None = None
                 ):
        self.__model = model
        self.__gpt_params = gpt_params or ChatGPTParams()

        self.base_instruction = base_instruction
        self.__generator = ChatGPTResponseGenerator(
            model=model,
            base_instruction=base_instruction,
            chat_completion_params=gpt_params,
            token_limit_exceed_handler=token_limit_exceed_handler,
            token_limit_tolerance=1024
        )

        self.__examples = examples
        self.__example_messages_cache: list[dict] | None = None

    @abstractmethod
    def _convert_input_to_message_content(self, input: InputType,
                                          params: ChatGPTFewShotParamsType | None = None) -> str:
        pass

    @abstractmethod
    def _postprocess_chatgpt_output(self, output: str, params: ChatGPTFewShotParamsType | None = None) -> OutputType:
        """

        :param output: A raw output string from the ChatCompletion call.
        :param params: An optional parameter for postprocessing. Passed from __get_example_messages.
        :return: A processed output.
        By raising an RegenerateRequestException, you can re-trigger the run logic.
        """
        pass

    def __get_example_messages(self, params: ChatGPTFewShotParamsType | None = None) -> list[ChatCompletionMessage] | None:
        if self.__examples is not None:
            if self.__example_messages_cache is None:
                self.__example_messages_cache = list(chain.from_iterable([[
                    ChatCompletionMessage(self._convert_input_to_message_content(sample, params),
                                          ChatCompletionMessageRole.SYSTEM, "example_user"),
                    ChatCompletionMessage(label, ChatCompletionMessageRole.SYSTEM, "example_assistant")
                ] for sample, label in self.__examples]))

            return self.__example_messages_cache
        else:
            return None

    async def run(self, input: InputType, params: ChatGPTFewShotParamsType | None = None) -> OutputType:
        self.__generator.initial_user_message = self.__get_example_messages(params)

        if params is not None and params.instruction_params is not None and isinstance(self.base_instruction, Template):
            self.__generator.base_instruction = self.base_instruction.render(**params.instruction_params)

        resp, _, _ = await self.__generator.get_response(
            [DialogueTurn(self._convert_input_to_message_content(input, params), True)])
        # print(resp)

        try:
            processed_resp = self._postprocess_chatgpt_output(resp, params)
            return processed_resp
        except RegenerateRequestException as ex:
            print(f"Regeneration requested due to an error - {ex.reason}")
            return await self.run(input, params)


class ChatGPTDialogueSummarizer(ChatGPTFewShotMapper[Dialogue, dict, ChatGPTDialogSummarizerParams]):

    def __init__(self, base_instruction: str | Template, model: str = ChatGPTModel.GPT_4_latest,
                 gpt_params: ChatGPTParams | None = None, examples: list[tuple[InputType, str]] | None = None,
                 dialogue_filter: Callable[[Dialogue, ChatGPTDialogSummarizerParams | None], Dialogue] | None = None,
                 token_limit_exceed_handler: TokenLimitExceedHandler | None = None
                 ):
        super().__init__(base_instruction, model, gpt_params, examples, token_limit_exceed_handler)
        self.dialogue_filter = dialogue_filter

    def _convert_input_to_message_content(self, input: Dialogue,
                                          params: ChatGPTDialogSummarizerParams | None = None) -> str:
        user_alias = (
            params.input_user_alias if params is not None and params.input_user_alias is not None else DEFAULT_USER_ALIAS)
        system_alias = (
            params.input_system_alias if params is not None and params.input_system_alias is not None else DEFAULT_SYSTEM_ALIAS)

        return DIALOGUE_TEMPLATE.render(user_alias=user_alias, system_alias=system_alias, dialogue=input)

    async def run(self, input: Dialogue, params: ChatGPTFewShotParamsType | None = None) -> dict:

        if self.dialogue_filter is not None:
            input = self.dialogue_filter(input, params)

        return await super().run(input, params)

    def _postprocess_chatgpt_output(self, output: str, params: ChatGPTDialogSummarizerParams | None = None) -> dict:
        try:
            return json.loads(output)
        except JSONDecodeError as ex:
            print(ex)
            raise RegenerateRequestException(f"Malformed JSON. Retry ChatCompletion. Original text: \n{output}")
