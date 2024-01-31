# https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-llama?tabs=azure-studio
from asyncio import to_thread
from enum import StrEnum
from functools import cache
from http.client import HTTPResponse
from typing import Any
from urllib import request, error, parse
import json
from transformers import AutoTokenizer

from chatlib.chat_completion import ChatCompletionAPI, ChatCompletionMessage, TokenLimitExceedError


class AzureLlama2Environment:
    _host: str | None = None
    _key: str | None = None

    ENDPOINT_CHAT_COMPLETION = '/v1/chat/completions'

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

    @cache
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
        return tokenizer

    def is_messages_within_token_limit(self, messages: list[ChatCompletionMessage], model: str,
                                       tolerance: int = 120) -> bool:
        return self.count_token_in_messages(messages, model) < 4096 + tolerance

    async def run_chat_completion(self, model: str, messages: list[ChatCompletionMessage], params: dict,
                                  trial_count: int = 5) -> Any:
        req = request.Request(AzureLlama2Environment.get_chat_completions_endpoint(), str.encode(json.dumps({
            "messages": [msg.to_dict() for msg in messages],
            **params
        })), AzureLlama2Environment.get_request_headers(), method="POST")

        trial = 0
        response: HTTPResponse | None = None
        while trial <= trial_count and response is None:
            try:
                response = await to_thread(request.urlopen, req)
            except (error.HTTPError, TokenLimitExceedError) as e:
                response = None
                trial += 1
                print("Azure Llama2 API error - ", e)
                print("Retry ChatCompletion.")

        result = None if response is None else json.loads(response.read())

        return result

    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.to_dict().items():
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
