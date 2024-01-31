from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import StrEnum
from typing import Optional, Any
from functools import cache


class ChatCompletionMessageRole(StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass(frozen=True)
class ChatCompletionMessage:
    content: str
    role: ChatCompletionMessageRole
    name: Optional[str] = None

    @cache
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class ChatCompletionAPI(ABC):

    @abstractmethod
    def authorize(self) -> bool:
        pass

    def _assert_authorize(self):
        assert self.authorize(), f"Authorization of {ChatCompletionAPI.__class__.__name__} required."

    @abstractmethod
    def is_messages_within_token_limit(self, messages: list[ChatCompletionMessage], model: str,
                                       tolerance: int = 120) -> bool:
        pass

    @abstractmethod
    async def _run_chat_completion_impl(self, model: str, messages: list[ChatCompletionMessage],
                                        params: dict,
                                        trial_count: int = 5) -> Any:
        pass

    async def run_chat_completion(self, model: str, messages: list[ChatCompletionMessage],
                                  params: dict,
                                  trial_count: int = 5) -> Any:
        self._assert_authorize()
        return await self._run_chat_completion_impl(model, messages, params, trial_count)

    @abstractmethod
    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        pass


class TokenLimitExceedError(Exception):
    pass
