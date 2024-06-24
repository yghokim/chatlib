from enum import StrEnum
from functools import cache
from typing import Any, Literal

from anthropic import Client, Anthropic, HUMAN_PROMPT, AI_PROMPT

from chatlib.llm.chat_completion_api import ChatCompletionAPI, ChatCompletionMessage, ChatCompletionResult, \
    ChatCompletionMessageRole, ChatCompletionFinishReason
from chatlib.utils.integration import APIAuthorizationVariableType, APIAuthorizationVariableSpec, \
    APIAuthorizationVariableSpecPresets


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


def convert_anthropic_stop_reason(reason: str | Literal["end_turn", "max_tokens", "stop_sequence"]) -> ChatCompletionFinishReason:
    if reason == "end_turn":
        return ChatCompletionFinishReason.Stop
    elif reason == 'max_tokens':
        return ChatCompletionFinishReason.Length
    else:
        return ChatCompletionFinishReason.Stop


class AnthropicModel(StrEnum):
    CLAUDE_21 = "claude-2.1"
    CLAUDE_3_OPUS_20240229 = "claude-3-opus-20240229"


# https://docs.anthropic.com/claude/reference/messages_post

class AnthropicChatCompletionAPI(ChatCompletionAPI):

    __api_key_spec = APIAuthorizationVariableSpecPresets.ApiKey

    @classmethod
    @cache
    def provider_name(cls) -> str:
        return "Anthropic"

    @classmethod
    def get_auth_variable_specs(cls) -> list[APIAuthorizationVariableSpec]:
        return [cls.__api_key_spec]

    @classmethod
    def _authorize_impl(cls, variables: dict[APIAuthorizationVariableSpec, Any]) -> bool:
        return True

    @property
    def __client(self) -> Client:
        return Anthropic(api_key=self.get_auth_variable_for_spec(self.__api_key_spec))


    def is_messages_within_token_limit(self, messages: list[ChatCompletionMessage], model: str,
                                       tolerance: int = 120) -> bool:
        return self.count_token_in_messages(messages, model) <= 200000 - tolerance

    async def _run_chat_completion_impl(self, model: str, messages: list[ChatCompletionMessage],
                                        params: dict) -> ChatCompletionResult:
        if len(messages) > 0 and messages[0].role is ChatCompletionMessageRole.SYSTEM:
            # Exists system prompt
            system_prompt = messages[0].content
            messages = messages[1:]
        else:
            system_prompt = None

        completion_result = self.__client.beta.messages.create(model=model,
                                                               system=system_prompt if system_prompt is not None else None,
                                                               messages=[msg.dict() for msg in messages],
                                                               max_tokens=1024,
                                                               **params,
                                                               )

        return ChatCompletionResult(
            message=ChatCompletionMessage(content=completion_result.content[0].text, role=ChatCompletionMessageRole.ASSISTANT),
            finish_reason=convert_anthropic_stop_reason(completion_result.stop_reason) if completion_result.stop_reason is not None else ChatCompletionFinishReason.Stop,
            model=model,
            provider=self.provider_name(),
            prompt_tokens=completion_result.usage.input_tokens,
            completion_tokens=completion_result.usage.output_tokens,
            total_tokens=completion_result.usage.input_tokens + completion_result.usage.output_tokens
        )

    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        self.assert_authorize()
        return self.__client.count_tokens(create_anthropic_prompt(messages))
