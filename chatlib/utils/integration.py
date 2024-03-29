import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from os import path, getcwd
from typing import Any

from dotenv import find_dotenv, set_key
from questionary import prompt
from stringcase import constcase

from chatlib.global_config import GlobalConfig
from chatlib.utils.validator import make_non_empty_string_validator
from chatlib.utils import env_helper


class APIAuthorizationVariableType(StrEnum):
    ApiKey = "api_key"
    Secret = "secret"
    Host = "host"
    Key = "key"


@dataclass(frozen=True)
class APIAuthorizationVariableSpec:
    variable_type: APIAuthorizationVariableType


class ServiceUnauthorizedError(Exception):
    pass


class IntegrationService(ABC):
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
