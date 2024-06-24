from enum import StrEnum
from functools import cache
from typing import Any

import tiktoken
from openai import AsyncOpenAI

from chatlib.llm.chat_completion_api import ChatCompletionMessage, ChatCompletionAPI, ChatCompletionResult, \
    ChatCompletionFinishReason
from chatlib.utils.integration import APIAuthorizationVariableType, APIAuthorizationVariableSpec, \
    APIAuthorizationVariableSpecPresets


class ChatGPTModel(StrEnum):
    GPT_3_5_latest = "gpt-3.5-turbo"
    GPT_3_5_16k_latest = "gpt-3.5-turbo-16k"
    GPT_3_5_0613 = "gpt-3.5-turbo-0613"
    GPT_3_5_1106 = "gpt-3.5-turbo-1106"
    GPT_3_5_0125 = "gpt-3.5-turbo-0125"
    GPT_4_latest = "gpt-4"
    GPT_4_32k_latest = "gpt-4-32k"
    GPT_4_0613 = "gpt-4-0613"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4_0125 = "gpt-4-0125-preview"
    GPT_4_1106 = "gpt-4-1106-preview"
    GPT_4o = "gpt-4o"


def get_token_limit(model: str):
    if model is ChatGPTModel.GPT_4_32k_latest:
        return 32000
    elif model is ChatGPTModel.GPT_3_5_16k_latest or model is ChatGPTModel.GPT_3_5_0125 or model is ChatGPTModel.GPT_3_5_1106:
        return 16000
    elif model is ChatGPTModel.GPT_4_0125 or model is ChatGPTModel.GPT_4_1106 or model is ChatGPTModel.GPT_4o:
        return 128000
    elif model.startswith("gpt-4-turbo"):
        return 128000
    elif model.startswith("gpt-3.5"):
        return 4096
    elif model.startswith("gpt-4"):
        return 8192
    else:
        raise NotImplementedError(f"token limit for model {model} is not implemented.")


class GPTChatCompletionAPI(ChatCompletionAPI):
    __api_key_spec = APIAuthorizationVariableSpecPresets.ApiKey

    @classmethod
    @cache
    def provider_name(self) -> str:
        return "Open AI"

    @classmethod
    def get_auth_variable_specs(cls) -> list[APIAuthorizationVariableSpec]:
        return [cls.__api_key_spec]

    @classmethod
    def _authorize_impl(cls, variables: dict[APIAuthorizationVariableSpec, Any]) -> bool:
        return True

    @property
    def __client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self.get_auth_variable_for_spec(self.__api_key_spec))

    def is_messages_within_token_limit(self, messages: list[ChatCompletionMessage], model: str,
                                       tolerance: int = 120) -> bool:
        return self.count_token_in_messages(messages, model) < get_token_limit(model) - tolerance

    async def _run_chat_completion_impl(self, model: str, messages: list[ChatCompletionMessage],
                                        params: dict) -> ChatCompletionResult:
        result = await self.__client.chat.completions.create(
            model=model,
            messages=[message.dict() for message in messages],
            **params
        )
        converted_result = ChatCompletionResult(
            message=ChatCompletionMessage(**result.choices[0].message.dict()),
            finish_reason=ChatCompletionFinishReason(result.choices[0].finish_reason),
            provider=self.provider_name(),
            model=result.model,
            **result.usage.dict()
        )

        return converted_result

    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        encoding = get_encoder_for_model(model)

        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4-0125-preview"
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            if self.config().verbose:
                print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self.count_token_in_messages(messages, model=ChatGPTModel.GPT_3_5_0613)
        elif "gpt-4-turbo-preview" in model:
            return self.count_token_in_messages(messages, model=ChatGPTModel.GPT_4_0125)
        elif "gpt-4" in model:
            if self.config().verbose:
                print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.count_token_in_messages(messages, model=ChatGPTModel.GPT_4_0613)
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.dict().items():
                try:
                    num_tokens += len(encoding.encode(value))
                except:
                    print(f"Error on token counting - {key}: {value}")

                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens


def get_encoder_for_model(model: ChatGPTModel | str) -> Any:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        return tiktoken.get_encoding("cl100k_base")
