from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import Any

from google.ai.generativelanguage_v1 import Candidate
from google.generativeai.types import GenerateContentResponse

from chatlib import env_helper
from chatlib.chat_completion import ChatCompletionAPI, ChatCompletionMessage, ChatCompletionMessageRole

import google.generativeai as genai


# https://ai.google.dev/tutorials/python_quickstart
# https://github.com/google/generative-ai-python/blob/main/google/generativeai/generative_models.py#L382-L423

class GeminiChatMessageRole(StrEnum):
    User = "user"
    Model = "model"


def convert_to_gemini_message(message: ChatCompletionMessage) -> dict:
    return dict(
        parts=[message.content],
        role=GeminiChatMessageRole.User if message.role is ChatCompletionMessageRole.USER or message.role is ChatCompletionMessageRole.SYSTEM else GeminiChatMessageRole.Model)


def convert_to_gemini_messages(messages: list[ChatCompletionMessage]) -> list[dict]:
    print("msg:", messages)
    return [convert_to_gemini_message(msg) for msg in messages]


def convert_candidate_to_choice(candidate: Candidate) -> dict:
    return {
        "message": {
            "content": candidate.content.parts[0].text
        },
        "finish_reason": "stop" if candidate.finish_reason == 1 else candidate.finish_reason
    }

class GeminiAPI(ChatCompletionAPI):

    def __init__(self):
        super().__init__()
        self.__model: genai.GenerativeModel | None = None
        self.__chat_model: genai.ChatSession | None = None

    def authorize(self) -> bool:
        api_key = env_helper.get_env_variable('GOOGLE_API_KEY')
        if api_key is not None:
            genai.configure(api_key=api_key)
            return True
        else:
            return False

    @cache
    def model(self) -> genai.GenerativeModel:
        return genai.GenerativeModel('gemini-pro')

    def is_messages_within_token_limit(self, messages: list[ChatCompletionMessage], model: str,
                                       tolerance: int = 120) -> bool:
        self._assert_authorize()
        # return self.count_token_in_messages(messages, model) < 4096 - tolerance
        return True

    async def _run_chat_completion_impl(self, model: str, messages: list[ChatCompletionMessage], params: dict,
                                        trial_count: int = 5) -> Any:
        injected_messages = [messages[0]] + [
            ChatCompletionMessage("Okay.", ChatCompletionMessageRole.ASSISTANT)] + messages[1:]

        converted_messages = convert_to_gemini_messages(injected_messages)
        print(converted_messages)
        response: GenerateContentResponse = self.model().generate_content(
            contents=converted_messages,
            generation_config=params
        )

        message = response.candidates[0].content.parts[0].text
        return {"choices": [convert_candidate_to_choice(candidate) for candidate in response.candidates],
                "usage": {}, "model": "gemini pro"}

    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        self._assert_authorize()
        return self.model().count_tokens(convert_to_gemini_messages(messages)).total_tokens
