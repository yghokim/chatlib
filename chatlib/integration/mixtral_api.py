from asyncio import to_thread
from enum import StrEnum
from typing import Any

from mistralai.client import MistralClient, MistralException
from mistralai.models.chat_completion import ChatMessage

from chatlib.chat_completion import ChatCompletionAPI, ChatCompletionMessage, APIAuthorizationVariableSpec, \
    APIAuthorizationVariableType


class MixtralModels(StrEnum):
    MistralTiny = "mistral-tiny"
    MistralMedium = "mistral-medium"


def _convert_to_mistral_chat_message(msg: ChatCompletionMessage) -> ChatMessage:
    return ChatMessage(
        content=msg.content,
        role=msg.role
    )


class MixtralAPI(ChatCompletionAPI):
    __api_key_spec = APIAuthorizationVariableSpec(APIAuthorizationVariableType.ApiKey)

    def __init__(self):
        super().__init__()
        self.__client: MistralClient | None = None

    @property
    def provider_name(self) -> str:
        return "Mistral AI"

    def get_auth_variable_specs(self) -> list[APIAuthorizationVariableSpec]:
        return [self.__api_key_spec]

    def _authorize_impl(self, variables: dict[APIAuthorizationVariableSpec, Any]) -> bool:
        api_key = variables[self.__api_key_spec]
        if api_key is not None:
            self.__client = MistralClient(api_key=variables[self.__api_key_spec])
            return True
        else:
            self.__client = None
            return False

    def is_messages_within_token_limit(self, messages: list[ChatCompletionMessage], model: str,
                                       tolerance: int = 120) -> bool:
        return True

    async def _run_chat_completion_impl(self, model: str, messages: list[ChatCompletionMessage], params: dict,
                                        trial_count: int = 5) -> Any:
        print("Run Mistral API...", messages)
        trial = 0
        result = None
        while trial <= trial_count and result is None:
            try:
                print([_convert_to_mistral_chat_message(msg) for msg in messages])
                result = await to_thread(self.__client.chat, model=model, messages=[_convert_to_mistral_chat_message(msg) for msg in messages])
            except (MistralException, Exception) as e:
                result = None
                trial += 1
                print("Mistral API error - ", e)
                print("Retry ChatCompletion.")
        return result

    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
        return 0
