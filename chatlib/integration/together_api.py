import json
from asyncio import to_thread
from enum import StrEnum
from functools import cache
from typing import Any
import requests

from chatlib.chat_completion import ChatCompletionAPI, ChatCompletionMessage, APIAuthorizationVariableSpec, \
    APIAuthorizationVariableType


class TogetherAIModels(StrEnum):
    Mixtral8x7BInstruct = "mistralai/Mixtral-8x7B-Instruct-v0.1"


class TogetherAPI(ChatCompletionAPI):

    __ENDPOINT = "https://api.together.xyz/v1/chat/completions"

    __api_key_spec = APIAuthorizationVariableSpec(APIAuthorizationVariableType.ApiKey)

    def __init__(self):
        self.__api_key : str | None = None

    @property
    @cache
    def provider_name(self) -> str:
        return "Together AI"

    def get_auth_variable_specs(self) -> list[APIAuthorizationVariableSpec]:
        return [self.__api_key_spec]

    def _authorize_impl(self, variables: dict[APIAuthorizationVariableSpec, Any]) -> bool:
        if self.__api_key_spec in variables and len(variables[self.__api_key_spec]) > 0:
            self.__api_key = variables[self.__api_key_spec]
            return True
        else:
            return False

    def is_messages_within_token_limit(self, messages: list[ChatCompletionMessage], model: str,
                                       tolerance: int = 120) -> bool:
        return True

    async def _run_chat_completion_impl(self, model: str, messages: list[ChatCompletionMessage], params: dict) -> Any:
        body = {
            "model": model,
            "n": 1,
            "messages": [msg.to_dict() for msg in messages]
        }

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.__api_key}"
        }

        response = await to_thread(requests.post, url=self.__ENDPOINT, json=body, headers=headers, data=None)
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            raise Exception(response.status_code, response.reason)

    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        return 0