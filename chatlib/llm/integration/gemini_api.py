from enum import StrEnum
from functools import cache
from typing import Any

import google.generativeai as genai
from google.ai.generativelanguage_v1 import Candidate
from google.generativeai.types import GenerateContentResponse

from chatlib.llm.chat_completion_api import ChatCompletionAPI, ChatCompletionMessage, ChatCompletionMessageRole
from chatlib.utils.integration import APIAuthorizationVariableType, APIAuthorizationVariableSpec, \
    APIAuthorizationVariableSpecPresets
from chatlib.llm.chat_completion_api import ChatCompletionResult

# https://ai.google.dev/tutorials/python_quickstart
# https://github.com/google/generative-ai-python/blob/main/google/generativeai/generative_models.py#L382-L423

_SAFETY_SETTINGS_BLOCK_NONE = [
    {
        "category": category,
        "threshold": "BLOCK_NONE"
    } for category in ["HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_HARASSMENT",
                       "HARM_CATEGORY_DANGEROUS_CONTENT"]
]

GEMINI_PRO_TOKEN_LIMIT = 30720


class GeminiChatMessageRole(StrEnum):
    User = "user"
    Model = "model"

    @classmethod
    def from_chat_completion_role(cls, role: ChatCompletionMessageRole) -> 'GeminiChatMessageRole':
        if role == ChatCompletionMessageRole.SYSTEM:
            return cls.User
        elif role == ChatCompletionMessageRole.ASSISTANT:
            return cls.Model
        elif role == ChatCompletionMessageRole.USER:
            return cls.User


def convert_to_gemini_message(message: ChatCompletionMessage) -> dict:
    return dict(
        parts=[message.content],
        role=GeminiChatMessageRole.from_chat_completion_role(message.role))


def convert_to_gemini_messages(messages: list[ChatCompletionMessage]) -> list[dict]:
    return [convert_to_gemini_message(msg) for msg in messages]


def convert_candidate_to_choice(candidate: Candidate) -> dict:

    role = None
    if candidate.content.role == GeminiChatMessageRole.User:
        role = ChatCompletionMessageRole.USER
    elif candidate.content.role == GeminiChatMessageRole.Model:
        role = ChatCompletionMessageRole.ASSISTANT

    return {
        "message": {
            "content": candidate.content.parts[0].text,
            "role": role
        },
        "token_count": candidate.token_count,
        "finish_reason": "stop" if candidate.finish_reason == 1 else candidate.finish_reason
    }


class GeminiAPI(ChatCompletionAPI):
    __api_key_spec = APIAuthorizationVariableSpecPresets.ApiKey

    @classmethod
    @cache
    def provider_name(cls) -> str:
        return "Google"

    @classmethod
    def get_auth_variable_specs(cls) -> list[APIAuthorizationVariableSpec]:
        return [cls.__api_key_spec]

    @classmethod
    def _authorize_impl(cls, variables: dict[APIAuthorizationVariableSpec, Any]) -> bool:
        genai.configure(api_key=variables[cls.__api_key_spec])
        return True

    def __init__(self,
                 safety_settings: list[dict] | None = None,
                 injected_initial_system_message: str = "Okay I will diligently follow that instruction."):
        super().__init__()
        self.__injected_initial_system_message = injected_initial_system_message
        self.__safety_settings = safety_settings or _SAFETY_SETTINGS_BLOCK_NONE

    @cache
    def model(self) -> genai.GenerativeModel:
        return genai.GenerativeModel('gemini-pro')

    def is_messages_within_token_limit(self, messages: list[ChatCompletionMessage], model: str,
                                       tolerance: int = 120) -> bool:
        self.assert_authorize()
        return self.count_token_in_messages(messages, model) < GEMINI_PRO_TOKEN_LIMIT - tolerance

    def __convert_messages(self, messages: list[ChatCompletionMessage]) -> list[ChatCompletionMessage]:
        # Tweak system instruction
        if len(messages) > 0 and messages[0].role is ChatCompletionMessageRole.SYSTEM:
            messages[0] = ChatCompletionMessage(
                content=f"<System instruction>\n{messages[0].content}\n</System instruction>",
                role=ChatCompletionMessageRole.SYSTEM
            )

        if len(messages) >= 2:
            if messages[0].role == ChatCompletionMessageRole.SYSTEM and messages[
                1].role is ChatCompletionMessageRole.ASSISTANT:
                return messages
            else:
                return [messages[0]] + [ChatCompletionMessage(content=self.__injected_initial_system_message,
                                                              role=ChatCompletionMessageRole.ASSISTANT)] + messages[1:]
        else:
            return messages + [
                ChatCompletionMessage(content=self.__injected_initial_system_message, role=ChatCompletionMessageRole.ASSISTANT),
                ChatCompletionMessage(content="Hi!", role=ChatCompletionMessageRole.USER)]

    async def _run_chat_completion_impl(self, model: str, messages: list[ChatCompletionMessage], params: dict) -> ChatCompletionResult:
        injected_messages = self.__convert_messages(messages)

        converted_messages = convert_to_gemini_messages(injected_messages)
        response: GenerateContentResponse = await self.model().generate_content_async(
            contents=converted_messages,
            generation_config=params,
            safety_settings=self.__safety_settings
        )

        top_choice = convert_candidate_to_choice(response.candidates[0])

        safety_ratings = {r.category: r.probability for r in response.prompt_feedback.safety_ratings}

        return ChatCompletionResult(
                message=ChatCompletionMessage(**top_choice["message"]),
                finish_reason=top_choice["finish_reason"],
                provider=self.provider_name(),
                model=model
            )

    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        self.assert_authorize()
        injected_messages = self.__convert_messages(messages)

        converted_messages = convert_to_gemini_messages(injected_messages)

        return self.model().count_tokens(converted_messages).total_tokens
