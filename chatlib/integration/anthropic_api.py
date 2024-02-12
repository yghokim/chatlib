from enum import StrEnum
from functools import cache
from typing import Any, Literal
from anthropic import Client, Anthropic, HUMAN_PROMPT, AI_PROMPT

from chatlib.chat_completion_api import ChatCompletionAPI, ChatCompletionMessage, ChatCompletionResult, \
    APIAuthorizationVariableSpec, APIAuthorizationVariableType, ChatCompletionMessageRole, ChatCompletionFinishReason


def create_anthropic_prompt(messages: list[ChatCompletionMessage]) -> str:
    if len(messages) > 0:
        prompt = ""
        if messages[0].role == ChatCompletionMessageRole.SYSTEM:
            prompt = messages[1].content
            messages_to_append = messages[1:]
        else:
            messages_to_append = messages

        for message in messages_to_append:
            prefix = None
            if message.role == ChatCompletionMessageRole.USER:
                prefix = HUMAN_PROMPT
            elif message.role == ChatCompletionMessageRole.SYSTEM:
                prefix = AI_PROMPT
            prompt += f"{prefix} {message.content}"

        prompt += f"{AI_PROMPT}"

        return prompt
    else:
        return f"{AI_PROMPT}"


def convert_anthropic_stop_reason(reason: str) -> ChatCompletionFinishReason:
    if reason == "end_turn":
        return ChatCompletionFinishReason.Stop
    elif reason == 'max_tokens':
        return ChatCompletionFinishReason.Length
    else:
        return ChatCompletionFinishReason.Stop


class AnthropicModels(StrEnum):
    CLAUDE_21 = "claude-2.1"

# https://docs.anthropic.com/claude/reference/messages_post

class AnthropicChatCompletionAPI(ChatCompletionAPI):
    def __init__(self):
        super().__init__()
        self.__client: Client | None = None

    __api_key_spec = APIAuthorizationVariableSpec(APIAuthorizationVariableType.ApiKey)

    @property
    @cache
    def provider_name(self) -> str:
        return "Anthropic"

    def get_auth_variable_specs(self) -> list[APIAuthorizationVariableSpec]:
        return [self.__api_key_spec]

    def _authorize_impl(self, variables: dict[APIAuthorizationVariableSpec, Any]) -> bool:
        if variables[self.__api_key_spec] is not None and len(variables[self.__api_key_spec]) > 0:
            if self.__client is not None:
                self.__client.api_key = variables[self.__api_key_spec]
            else:
                self.__client = Anthropic(api_key=variables[self.__api_key_spec])
            return True
        else:
            return False

    def is_messages_within_token_limit(self, messages: list[ChatCompletionMessage], model: str,
                                       tolerance: int = 120) -> bool:
        return self.count_token_in_messages(messages, model) <= 200000 - tolerance

    async def _run_chat_completion_impl(self, model: str, messages: list[ChatCompletionMessage],
                                        params: dict) -> ChatCompletionResult:
        completion_result = self.__client.completions.create(model=model,
                                                             prompt=create_anthropic_prompt(messages),
                                                             **params,
                                                             max_tokens_to_sample=1000
                                                             )

        return ChatCompletionResult(
            message=ChatCompletionMessage(completion_result.completion, role=ChatCompletionMessageRole.ASSISTANT),
            finish_reason=convert_anthropic_stop_reason(completion_result.stop_reason),
            model=model,
            provider=self.provider_name
        )

    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        self.assert_authorize()
        return self.__client.count_tokens(create_anthropic_prompt(messages))
