import json
from asyncio import to_thread
from enum import StrEnum
from functools import cache
from typing import Any

import requests

from chatlib.llm.chat_completion_api import ChatCompletionAPI, ChatCompletionMessage, ChatCompletionResult, ChatCompletionFinishReason
from chatlib.utils.integration import APIAuthorizationVariableSpec, \
    APIAuthorizationVariableSpecPresets


class TogetherAIModel(StrEnum):
    Mixtral8x7BInstruct = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    Vicuna13B1_5 = "lmsys/vicuna-13b-v1.5"


class TogetherAPI(ChatCompletionAPI):
    __ENDPOINT = "https://api.together.xyz/v1/chat/completions"

    __api_key_spec = APIAuthorizationVariableSpecPresets.ApiKey

    @classmethod
    @cache
    def provider_name(cls) -> str:
        return "Together AI"

    @classmethod
    def get_auth_variable_specs(cls) -> list[APIAuthorizationVariableSpec]:
        return [cls.__api_key_spec]

    @classmethod
    def _authorize_impl(cls, variables: dict[APIAuthorizationVariableSpec, Any]) -> bool:
        return True

    def is_messages_within_token_limit(self, messages: list[ChatCompletionMessage], model: str,
                                       tolerance: int = 120) -> bool:
        return True

    async def _run_chat_completion_impl(self, model: str, messages: list[ChatCompletionMessage], params: dict) -> Any:
        body = {
            "model": model,
            "n": 1,
            "stream": False,
            "messages": [msg.dict() for msg in messages]
        }

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.get_auth_variable_for_spec(self.__api_key_spec)}"
        }

        response = await to_thread(requests.post, url=self.__ENDPOINT, json=body, headers=headers, data=None)
        if response.status_code == 200:
            json_response = json.loads(response.text)
            print(json_response)
            return ChatCompletionResult(
                message=ChatCompletionMessage(**json_response["choices"][0]["message"]),
                finish_reason=ChatCompletionFinishReason.Stop,
                provider=self.provider_name(),
                model=json_response["model"],
                **json_response["usage"]
            )
        else:
            raise Exception(response.status_code, response.reason)

    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        return 0
