import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import StrEnum
from functools import cache
from os import path, getcwd
from typing import Optional, Any, Callable

from dotenv import find_dotenv, set_key
from pydantic import BaseModel, ConfigDict, Field
from questionary import prompt
from stringcase import constcase

from chatlib import env_helper
from chatlib.global_config import GlobalConfig


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


class APIAuthorizationVariableType(StrEnum):
    ApiKey = "api_key"
    Secret = "secret"
    Host = "host"
    Key = "key"


@dataclass(frozen=True)
class APIAuthorizationVariableSpec:
    variable_type: APIAuthorizationVariableType


class TokenLimitExceedError(Exception):
    pass


class ServiceUnauthorizedError(Exception):
    pass


@dataclass(frozen=True)
class ChatCompletionRetryRequestedException(Exception):
    caused_by: Exception | None = None


def make_non_empty_string_validator(msg: str) -> Callable:
    return lambda text: True if len(text.strip()) > 0 else msg


class ChatCompletionAPI(ABC):

    @classmethod
    @abstractmethod
    def provider_name(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def get_auth_variable_specs(cls) -> list[APIAuthorizationVariableSpec]:
        pass

    @classmethod
    @cache
    def env_key_for_spec(cls, spec: APIAuthorizationVariableSpec) -> str:
        return re.sub(r'_+', '_', constcase(cls.provider_name() + "_" + spec.variable_type))

    @classmethod
    @cache
    def get_auth_variable_for_spec(cls, spec: APIAuthorizationVariableSpec) -> str:
        return env_helper.get_env_variable(cls.env_key_for_spec(spec))

    @classmethod
    def authorize(cls) -> bool:
        variables: dict[APIAuthorizationVariableSpec, Any] = {}
        for spec in cls.get_auth_variable_specs():
            var = cls.get_auth_variable_for_spec(spec)
            if var is not None:
                variables[spec] = var
            else:
                cls.get_auth_variable_for_spec.cache_clear()
                return False

        return cls._authorize_impl(variables)

    @classmethod
    @abstractmethod
    def _authorize_impl(cls, variables: dict[APIAuthorizationVariableSpec, Any]) -> bool:
        pass

    @classmethod
    def assert_authorize(cls):
        if cls.authorize():
            return
        elif GlobalConfig.is_cli_mode:  # If on a CLI, authorize directly.
            cls._request_auth_variables_cli()
            cls.assert_authorize()
        else:
            raise ServiceUnauthorizedError(cls.provider_name())

    @classmethod
    def _request_auth_variables_cli(cls):
        questions = []
        for spec in cls.get_auth_variable_specs():
            default_question_spec = {
                "type": 'text',
                "name": cls.env_key_for_spec(spec)
            }

            if spec.variable_type is APIAuthorizationVariableType.ApiKey:
                default_question_spec.update({
                    "message": f'Please enter your API key for {cls.provider_name()}:',
                    "validate": make_non_empty_string_validator("Please enter a valid API key.")})

            elif spec.variable_type is APIAuthorizationVariableType.Key:
                default_question_spec.update({
                    "message": f'Please enter your key for {cls.provider_name()}:',
                    "validate": make_non_empty_string_validator("Please enter a valid key.")
                })
            elif spec.variable_type is APIAuthorizationVariableType.Host:
                default_question_spec.update({
                    "message": f'Please enter a host address for {cls.provider_name()}:',
                    "validate": make_non_empty_string_validator("Please enter a valid address.")
                })

            questions.append(default_question_spec)

        answers = prompt(questions)

        env_file = find_dotenv(usecwd=True)
        if not path.exists(env_file):
            env_file = open(path.join(getcwd(), '.env'), 'w', encoding='utf-8')
            env_file.close()
            env_file = find_dotenv(usecwd=True)

        for env_key, env_value in answers.items():
            set_key(env_file, env_key, env_value)
        cls.get_auth_variable_for_spec.cache_clear()

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
                result = await self._run_chat_completion_impl(model, messages, params)
            except ChatCompletionRetryRequestedException as e:
                result = None
                trial += 1
                print(f"Retry chat completion of {self.provider_name} - {e.caused_by}")

        return result

    @abstractmethod
    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        pass
