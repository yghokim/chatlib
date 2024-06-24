# https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-llama?tabs=azure-studio
import json
from asyncio import to_thread
from enum import StrEnum
from functools import cache
from http import HTTPStatus
from http.client import HTTPResponse
from typing import Any
from urllib import request, error, parse

from transformers import AutoTokenizer

from chatlib.llm.chat_completion_api import ChatCompletionAPI, ChatCompletionMessage, TokenLimitExceedError, \
    ChatCompletionRetryRequestedException, \
    ChatCompletionResult
from chatlib.utils.integration import APIAuthorizationVariableType, APIAuthorizationVariableSpec, \
    APIAuthorizationVariableSpecPresets


class AzureLlama2Environment:
    _host: str | None = None
    _key: str | None = None

    ENDPOINT_CHAT_COMPLETION = '/v1/chat/completions'

    @classmethod
    def is_authorized(cls) -> bool:
        return cls._host is not None and cls._key is not None

    @classmethod
    def get_host(cls: 'AzureLlama2Environment') -> str | None:
        return cls._host

    @classmethod
    def set_host(cls: 'AzureLlama2Environment', host: str):
        if cls._host != host:
            cls.get_chat_completions_endpoint.cache_clear()
        cls._host = host

    @classmethod
    def get_key(cls: 'AzureLlama2Environment') -> str | None:
        return cls._key

    @classmethod
    def set_key(cls: 'AzureLlama2Environment', new_key: str):
        if cls._key != new_key:
            cls.get_request_headers.cache_clear()
        cls._key = new_key

    @classmethod
    def set_credentials(cls: 'AzureLlama2Environment', host: str, key: str):
        cls.set_host(host)
        cls.set_key(key)

    @classmethod
    @cache
    def get_chat_completions_endpoint(cls) -> str:
        return parse.urljoin(cls._host, cls.ENDPOINT_CHAT_COMPLETION)

    @classmethod
    @cache
    def get_request_headers(cls) -> dict:
        return {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + cls._key)}


class Llama2Model(StrEnum):
    Llama2_70b_chat = "Llama2_70b_chat"


class AzureLlama2ChatCompletionAPI(ChatCompletionAPI):
    __host_spec = APIAuthorizationVariableSpecPresets.Host
    __key_spec = APIAuthorizationVariableSpecPresets.Key

    @classmethod
    @cache
    def provider_name(cls) -> str:
        return "Azure Llama2"

    @classmethod
    def get_auth_variable_specs(cls) -> list[APIAuthorizationVariableSpec]:
        return [cls.__host_spec, cls.__key_spec]

    @classmethod
    def _authorize_impl(cls, variables: dict[APIAuthorizationVariableSpec, Any]) -> bool:
        AzureLlama2Environment.set_host(variables[cls.__host_spec])
        AzureLlama2Environment.set_key(variables[cls.__key_spec])
        return True

    @cache
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
        return tokenizer

    def is_messages_within_token_limit(self, messages: list[ChatCompletionMessage], model: str,
                                       tolerance: int = 120) -> bool:
        return self.count_token_in_messages(messages, model) < 4096 + tolerance

    async def _run_chat_completion_impl(self, model: str, messages: list[ChatCompletionMessage], params: dict) -> Any:
        req = request.Request(AzureLlama2Environment.get_chat_completions_endpoint(), str.encode(json.dumps({
            "messages": [msg.dict() for msg in messages],
            **params
        })), AzureLlama2Environment.get_request_headers(), method="POST")

        try:
            response: HTTPResponse = await to_thread(request.urlopen, url=req)
            if response.status == HTTPStatus.OK:
                json_response = json.loads(response.read())
                return ChatCompletionResult(
                    message=ChatCompletionMessage(**json_response["choices"][0]["message"]),
                    finish_reason=json_response["choices"][0]["finish_reason"],
                    provider=self.provider_name(),
                    model=model,
                    **json_response["usage"]
                )
            else:
                raise ChatCompletionRetryRequestedException()
        except (error.HTTPError, TokenLimitExceedError) as e:
            raise ChatCompletionRetryRequestedException(e) from e

    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.dict().items():
                try:
                    num_tokens += len(self.get_tokenizer().encode(value))
                except Exception as e:
                    print(e)
                    print(f"Error on token counting - {key}: {value}")

                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        print("Estimated token count: ", num_tokens)
        return num_tokens
