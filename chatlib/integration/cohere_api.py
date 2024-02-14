from functools import cache
from typing import Any

from cohere import AsyncClient

from chatlib.chat_completion_api import ChatCompletionAPI, ChatCompletionMessage, ChatCompletionResult, \
    APIAuthorizationVariableSpec, APIAuthorizationVariableType


class CohereChatAPI(ChatCompletionAPI):

    @classmethod
    @cache
    def provider_name(cls) -> str:
        return "Cohere"

    @classmethod
    def get_auth_variable_specs(cls) -> list[APIAuthorizationVariableSpec]:
        return [cls.__api_key_spec]

    @classmethod
    def _authorize_impl(cls, variables: dict[APIAuthorizationVariableSpec, Any]) -> bool:
        return True

    @property
    def __client(self)->AsyncClient:
        return AsyncClient(api_key=self._get_auth_variable_for_spec())

    def is_messages_within_token_limit(self, messages: list[ChatCompletionMessage], model: str,
                                       tolerance: int = 120) -> bool:
        pass

    async def _run_chat_completion_impl(self, model: str, messages: list[ChatCompletionMessage],
                                        params: dict) -> ChatCompletionResult:
        pass

    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        pass

    __api_key_spec = APIAuthorizationVariableSpec(APIAuthorizationVariableType.ApiKey)
