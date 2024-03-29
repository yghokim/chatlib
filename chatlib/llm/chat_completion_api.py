from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from chatlib.utils.integration import IntegrationService


class ChatCompletionMessageRole(StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ChatCompletionFunction:
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatCompletionToolCall:
    index: int
    id: str
    function: ChatCompletionFunction
    type: str = "function"


class ChatCompletionMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str | None

    role: ChatCompletionMessageRole
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: list[ChatCompletionToolCall] | None = None

    @cache
    def dict(self) -> dict:
        return super().dict(exclude_none=True)


class ChatCompletionFinishReason(StrEnum):
    Stop = "stop"
    Length = "length"
    Tool = "tool_calls"
    ContentFilter = "content_filter"


class ChatCompletionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: ChatCompletionMessage
    finish_reason: ChatCompletionFinishReason

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)

    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    total_tokens: int | None = None


class TokenLimitExceedError(Exception):
    pass


@dataclass(frozen=True)
class ChatCompletionRetryRequestedException(Exception):
    caused_by: Exception | None = None


class ChatCompletionAPIGlobalConfig(BaseModel):
    verbose: bool | None = False


class ChatCompletionAPI(IntegrationService, ABC):

    def __init__(self):
        self.__config = ChatCompletionAPIGlobalConfig(verbose=False)

    def config(self) -> ChatCompletionAPIGlobalConfig:
        return self.__config

    @abstractmethod
    def is_messages_within_token_limit(self, messages: list[ChatCompletionMessage], model: str,
                                       tolerance: int = 120) -> bool:
        pass

    @abstractmethod
    async def _run_chat_completion_impl(self, model: str, messages: list[ChatCompletionMessage],
                                        params: dict) -> ChatCompletionResult:
        pass

    async def run_chat_completion(self, model: str, messages: list[ChatCompletionMessage],
                                  params: dict,
                                  trial_count: int = 5) -> ChatCompletionResult | None:
        self.assert_authorize()
        trial = 0
        result = None
        while trial <= trial_count and result is None:
            try:
                if self.config().verbose:
                    print(f"Run chat completion on {model} with messages:", messages)

                result = await self._run_chat_completion_impl(model, messages, params)
            except ChatCompletionRetryRequestedException as e:
                result = None
                trial += 1

                if self.config().verbose:
                    print(f"Retry chat completion of {self.provider_name} - {e.caused_by}")

        return result

    @abstractmethod
    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        pass
