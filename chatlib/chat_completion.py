from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import StrEnum
from os import path, getcwd
from typing import Optional, Any, Callable
from functools import cache
from questionary import prompt
from stringcase import constcase

from chatlib import env_helper

from dotenv import find_dotenv, set_key

from global_config import GlobalConfig


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


class TokenLimitExceedError(Exception):
    pass

class ServiceUnauthorizedError(Exception):
    pass


def make_non_empty_string_validator(msg: str) -> Callable:
    return lambda text: True if len(text.strip()) > 0 else msg

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

    def assert_authorize(self):
        if self.authorize():
            return
        elif GlobalConfig.is_cli_mode: # If on a CLI, authorize directly.
            self._request_auth_variables_cli()
            self.assert_authorize()
        else:
            raise ServiceUnauthorizedError(self.provider_name)

    def _request_auth_variables_cli(self):
        questions = []
        for spec in self.get_auth_variable_specs():
            default_question_spec = {
                    "type": 'text',
                    "name": self.__env_key_for_spec(spec)
            }

            if spec.variable_type is APIAuthorizationVariableType.ApiKey:
                default_question_spec.update({
                    "message": f'Please enter your API key for {self.provider_name}:',
                    "validate": make_non_empty_string_validator("Please enter a valid API key.")})

            elif spec.variable_type is APIAuthorizationVariableType.Key:
                default_question_spec.update({
                    "type": 'text',
                    "name": spec.variable_type,
                    "message": f'Please enter your key for {self.provider_name}:',
                    "validate": make_non_empty_string_validator("Please enter a valid key.")
                })
            elif spec.variable_type is APIAuthorizationVariableType.Host:
                default_question_spec.update({
                    "type": 'text',
                    "name": self.__env_key_for_spec(spec),
                    "message": f'Please enter a host address for {self.provider_name}:',
                    "validate": make_non_empty_string_validator("Please enter a valid address.")
                })

            questions.append(default_question_spec)

        answers = prompt(questions)

        env_file = find_dotenv()
        if not path.exists(env_file):
            env_file = open(path.join(getcwd(), '.env'), 'w', encoding='utf-8')
            env_file.close()
            env_file = find_dotenv()

        for env_key, env_value in answers.items():
            set_key(env_file, env_key, env_value)

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
        self.assert_authorize()
        return await self._run_chat_completion_impl(model, messages, params, trial_count)

    @abstractmethod
    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        pass

