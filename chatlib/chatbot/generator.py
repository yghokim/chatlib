import json
from abc import ABC, abstractmethod
from time import perf_counter
from typing import TypeAlias, Callable, Awaitable, Any

from jinja2 import Template

from .types import Dialogue, RegenerateRequestException
from .. import dict_utils
from ..chat_completion import ChatCompletionMessage, ChatCompletionAPI, ChatCompletionMessageRole, TokenLimitExceedError
from ..message_transformer import MessageTransformerChain, run_message_transformer_chain, \
    SpecialTokenListExtractionTransformer


class ResponseGenerator(ABC):

    def __init__(self,
                 message_transformers: MessageTransformerChain | None = None):
        self._message_transformers = message_transformers

    async def initialize(self):
        pass

    def _pre_get_response(self, dialog: Dialogue):
        pass

    @abstractmethod
    async def _get_response_impl(self, dialog: Dialogue, dry: bool = False) -> tuple[str, dict | None]:
        pass

    async def get_response(self, dialog: Dialogue, dry: bool = False) -> tuple[str, dict | None, int]:
        start = perf_counter()

        try:
            self._pre_get_response(dialog)
            response, metadata = await self._get_response_impl(dialog, dry)
        except RegenerateRequestException as regen:
            print(f"Regenerate response. Reason: {regen.reason}")
            response, metadata = await self._get_response_impl(dialog, dry)
        except Exception as ex:
            raise ex

        if self._message_transformers is not None:
            cleaned_response, metadata = run_message_transformer_chain(response, metadata, self._message_transformers)
            if cleaned_response != response:
                metadata = dict_utils.set_nested_value(metadata, "original_message", response)
                response = cleaned_response

        end = perf_counter()

        return response, metadata, int((end - start) * 1000)

    @abstractmethod
    def write_to_json(self, parcel: dict):
        pass

    @abstractmethod
    def restore_from_json(self, parcel: dict):
        pass


TokenLimitExceedHandler: TypeAlias = Callable[[Dialogue, list[ChatCompletionMessage]], Awaitable[Any]]


class ChatCompletionResponseGenerator(ResponseGenerator):

    def __init__(self,
                 api: ChatCompletionAPI,
                 model: str,
                 base_instruction: str | Template | None = None,
                 instruction_parameters: dict | None = None,
                 initial_user_message: str | list[ChatCompletionMessage] | None = None,
                 chat_completion_params: dict | None = None,
                 function_handler: Callable[[str, dict | None], Awaitable[Any]] | None = None,
                 special_tokens: list[tuple[str, str, Any]] | None = None, verbose: bool = False,

                 token_limit_exceed_handler: TokenLimitExceedHandler | None = None,
                 token_limit_tolerance: int = 1024
                 ):

        self.__api = api

        self.model = model

        self.__params = chat_completion_params or {}

        self.initial_user_message = initial_user_message

        self.__base_instruction = base_instruction if base_instruction is not None else "You are a ChatGPT assistant that is empathetic and supportive."

        self.__instruction_parameters = instruction_parameters

        self.__resolve_instruction()

        self.function_handler = function_handler

        self.verbose = verbose

        self.__token_limit_exceed_handler = token_limit_exceed_handler
        self.__token_limit_tolerance = token_limit_tolerance

        if special_tokens is not None and len(special_tokens) > 0:

            def onTokenFound(tokens: list[str], original_message: str, cleaned_message: str, metadata: dict | None):
                for token in tokens:
                    key, value = [(k, v) for tok, k, v in special_tokens if tok == token][0]
                    metadata = dict_utils.set_nested_value(metadata, key, value)
                metadata = dict_utils.set_nested_value(metadata, ["chatcompletion", "token_uncleaned_message"],
                                                       original_message)
                return cleaned_message, metadata

            transformer = SpecialTokenListExtractionTransformer("special_tokens", [tok for tok, k, v in special_tokens],
                                                                onTokenFound)

            super().__init__(message_transformers=[transformer])
        else:
            super().__init__()

    def __resolve_instruction(self):
        if isinstance(self.__base_instruction, Template):
            if self.__instruction_parameters is not None:
                self.__instruction = self.__base_instruction.render(**self.__instruction_parameters)
                self._on_instruction_updated(self.__instruction_parameters)
            else:
                self.__instruction = self.__base_instruction.render()
        else:
            self.__instruction = self.__base_instruction

    def _on_instruction_updated(self, params: dict):
        pass

    @property
    def base_instruction(self) -> str:
        return self.__base_instruction

    @base_instruction.setter
    def base_instruction(self, new: str):
        self.__base_instruction = new
        self.__resolve_instruction()

    @property
    def _instruction_parameters(self) -> dict:
        return self.__instruction_parameters

    def update_instruction_parameters(self, params: dict):
        if self.__instruction_parameters is not None:
            self.__instruction_parameters.update(params)
        else:
            self.__instruction_parameters = params
        self.__resolve_instruction()

    async def retrieve_response_with_function_result(self, function_name: str, messages: list[ChatCompletionMessage],
                                                     function_messages: list[ChatCompletionMessage],
                                                     function_call_result: str) -> any:
        return await self.__api.run_chat_completion(self.model, messages + function_messages, self.__params)

    async def _get_response_impl(self, dialog: Dialogue, dry: bool = False) -> tuple[str, dict | None]:
        dialogue_converted: list[ChatCompletionMessage] = []
        for turn in dialog:
            function_messages = dict_utils.get_nested_value(turn.metadata, ["chatcompletion", "function_messages"])
            if function_messages is not None:
                dialogue_converted.extend(turn.metadata["chatcompletion"]["function_messages"])

            original_message = dict_utils.get_nested_value(turn.metadata, ["chatcompletion", "token_uncleaned_message"])
            dialogue_converted.append(
                ChatCompletionMessage(original_message if original_message is not None else turn.message,
                                      ChatCompletionMessageRole.USER if turn.is_user else ChatCompletionMessageRole.ASSISTANT))

        instruction = self.__instruction
        if instruction is not None:

            instruction_turn = ChatCompletionMessage(instruction, ChatCompletionMessageRole.SYSTEM)

            messages = [instruction_turn]
            if self.initial_user_message is not None:
                if isinstance(self.initial_user_message, str):
                    messages.append(ChatCompletionMessage(self.initial_user_message, ChatCompletionMessageRole.USER))
                else:
                    messages.extend(self.initial_user_message)

            messages.extend(dialogue_converted)
        else:
            messages = dialogue_converted

        if self.__api.is_messages_within_token_limit(messages, self.model, self.__token_limit_tolerance):
            result = await self.__api.run_chat_completion(self.model, messages, self.__params)
        else:
            print(f"Token overflow - {len(messages)} message(s).")
            if self.__token_limit_exceed_handler is not None:
                result = await self.__token_limit_exceed_handler(dialog, messages)
            else:
                raise TokenLimitExceedError()

        top_choice = result["choices"][0]

        base_metadata = {"chatcompletion": {
            "usage": dict(**result["usage"]),
            "model": result["model"]
        }}

        if top_choice["finish_reason"] == 'stop':
            response_text = top_choice["message"]["content"]
            return response_text, base_metadata
        elif top_choice["finish_reason"] == 'function_call':
            function_call_info = top_choice["message"]["function_call"]
            function_name = function_call_info["name"]
            function_args = json.loads(function_call_info["arguments"])

            if self.verbose: print(f"Call function - {function_name} ({function_args})")

            function_call_result = await self.function_handler(function_name, function_args)
            function_turn = ChatCompletionMessage(function_call_result, ChatCompletionMessageRole.FUNCTION,
                                                  name=function_name)
            function_messages = [top_choice["message"], function_turn]

            new_result = await self.retrieve_response_with_function_result(function_name, messages, function_messages,
                                                                           function_call_result)

            top_choice = new_result.choices[0]
            if top_choice["finish_reason"] == 'stop':
                response_text = top_choice["message"]["content"]
                return response_text, dict_utils.set_nested_value(base_metadata,
                                                                  ["chatcompletion", "function_messages"],
                                                                  function_messages)
            else:
                print("Shouldn't reach here")

        else:
            raise Exception(f"ChatGPT error - {top_choice['finish_reason']}")

    def write_to_json(self, parcel: dict):
        parcel["model"] = self.model
        parcel["params"] = self.__params
        parcel["initial_user_message"] = self.initial_user_message
        parcel["base_instruction"] = self.__base_instruction
        parcel["instruction_parameters"] = self.__instruction_parameters
        parcel["verbose"] = self.verbose

    def restore_from_json(self, parcel: dict):
        self.model = parcel["model"]
        self.__params = parcel["params"]
        self.initial_user_message = parcel["initial_user_message"]
        self.__base_instruction = parcel["base_instruction"]
        self.__instruction_parameters = parcel["instruction_parameters"]
        self.verbose = parcel["verbose"]
        self.__resolve_instruction()
