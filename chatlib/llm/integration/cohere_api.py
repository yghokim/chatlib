from enum import StrEnum
from functools import cache
from typing import Any

from cohere import AsyncClient

from chatlib.llm.chat_completion_api import ChatCompletionAPI, ChatCompletionMessage, ChatCompletionResult, \
    ChatCompletionMessageRole, ChatCompletionFinishReason
from chatlib.utils.integration import APIAuthorizationVariableType, APIAuthorizationVariableSpec, \
    APIAuthorizationVariableSpecPresets


class CohereModel(StrEnum):
    Command = "command"
    CommandNightly = "command-nightly"


class CohereChatRole(StrEnum):
    User = "USER",
    Assistant = "CHATBOT"


def _convert_to_cohere_chat_role(role: ChatCompletionMessageRole) -> CohereChatRole:
    if role is ChatCompletionMessageRole.SYSTEM or role is ChatCompletionMessageRole.USER:
        return CohereChatRole.User
    elif role is ChatCompletionMessageRole.ASSISTANT:
        return CohereChatRole.Assistant
    else:
        raise ValueError(f"Not compatible with Cohere roles - {role}")


def _convert_to_cohere_message(message: ChatCompletionMessage) -> dict:
    return {
        "role": _convert_to_cohere_chat_role(message.role),
        "message": message.content
    }


# https://docs.cohere.com/reference/chat

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
    def __client(self) -> AsyncClient:
        return AsyncClient(api_key=self.get_auth_variable_for_spec(self.__api_key_spec))

    def is_messages_within_token_limit(self, messages: list[ChatCompletionMessage], model: str,
                                       tolerance: int = 120) -> bool:
        return True

    async def _run_chat_completion_impl(self, model: str, messages: list[ChatCompletionMessage],
                                        params: dict) -> ChatCompletionResult:
        response = await self.__client.chat(chat_history=[_convert_to_cohere_message(msg) for msg in messages[:-1]],
                                            message=messages[-1].content,
                                            model=model,
                                            **params
                                            )

        return ChatCompletionResult(
            message=ChatCompletionMessage(content=response.text, role=ChatCompletionMessageRole.ASSISTANT),
            finish_reason=ChatCompletionFinishReason.Stop,
            provider=self.provider_name(),
            model=model,
            prompt_tokens=response.token_count['prompt_tokens'],
            completion_tokens=response.token_count['response_tokens'],
            total_tokens=response.token_count['total_tokens']
        )

    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        pass

    __api_key_spec = APIAuthorizationVariableSpecPresets.ApiKey
