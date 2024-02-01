from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import StrEnum
from typing import Optional, Any
from functools import cache

from stringcase import constcase

from chatlib import env_helper


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


class APIAuthorizationVariableType(StrEnum):
    ApiKey = "api_key"
    Secret = "secret"
    Host = "host"
    Key = "key"


@dataclass(frozen=True)
class APIAuthorizationVariableSpec:
    variable_type: APIAuthorizationVariableType


class ChatCompletionAPI(ABC):

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    def get_auth_variable_specs(self) -> list[APIAuthorizationVariableSpec]:
        pass

    def __env_key_for_spec(self, spec: APIAuthorizationVariableSpec) -> str:
        return constcase(self.provider_name + "_" + spec.variable_type)

    def authorize(self) -> bool:
        variables: dict[APIAuthorizationVariableSpec, Any] = {}
        for spec in self.get_auth_variable_specs():
            var = env_helper.get_env_variable(self.__env_key_for_spec(spec))
            if var is not None:
                variables[spec] = var
            else:
                return False

        return self._authorize_impl(variables)

    @abstractmethod
    def _authorize_impl(self, variables: dict[APIAuthorizationVariableSpec, Any]) -> bool:
        pass

    def _assert_authorize(self):
        assert self.authorize(), f"Authorization of {self.provider_name} required."

    def _request_auth_variables_cli(self):
        pass

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
