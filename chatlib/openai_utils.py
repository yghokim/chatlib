from asyncio import to_thread
from enum import StrEnum
from typing import TypedDict, Optional, Any, Callable

import openai
import tiktoken


class ChatGPTModel(StrEnum):
    GPT_3_5_latest = "gpt-3.5-turbo"
    GPT_3_5_16k_latest = "gpt-3.5-turbo-16k"
    GPT_4_latest = "gpt-4"
    GPT_4_32k_latest = "gpt-4-32k"
    GPT_4_0613 = "gpt-4-0613"
    GPT_3_5_0613 = "gpt-3.5-0613"


class ChatGPTRole(StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class ChatGPTFunctionParameterProperty(TypedDict):
    type: str
    description: Optional[str]
    enum: Optional[list[str]]


class ChatGPTFunctionParameters(TypedDict):
    type: str
    properties: dict[str, ChatGPTFunctionParameterProperty]


class ChatGPTFunctionInfo(TypedDict):
    name: str
    description: Optional[str]
    parameters: ChatGPTFunctionParameters


class ChatGPTParams:
    def __init__(self,
                 temperature: float | None = None,
                 presence_penalty: float | None = None,
                 frequency_penalty: float | None = None,
                 functions: list[ChatGPTFunctionInfo | dict] | None = None
                 ):
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.functions = functions

    def to_params(self) -> dict:
        return {key: value for key, value in self.__dict__.items() if value is not None}


def get_token_limit(model: ChatGPTModel):
    if model is ChatGPTModel.GPT_4_32k_latest:
        return 32000
    elif model is ChatGPTModel.GPT_3_5_16k_latest:
        return 16000
    elif model.startswith("gpt-3.5"):
        return 4096
    elif model.startswith("gpt-4"):
        return 8192
    else:
        raise NotImplementedError(f"token limit for model {model} is not implemented.")


def is_messages_within_token_limit(messages: list[dict], model: ChatGPTModel, tolerance: int = 120) -> bool:
    return count_token_in_messages(messages, model) < get_token_limit(model) - tolerance


def make_chat_completion_message(message: str, role: str, name: str = None) -> dict:
    result = {
        "content": message,
        "role": role
    }

    if name is not None and len(name) > 0:
        result["name"] = name

    return result


async def run_chat_completion(model: str, messages: list[dict], gpt_params: ChatGPTParams,
                              trial_count: int = 5) -> Any:
    trial = 0
    result = None
    while trial <= trial_count and result is None:
        try:
            result = await to_thread(openai.ChatCompletion.create,
                                     model=model,
                                     messages=messages,
                                     **gpt_params.to_params()
                                     )
        except (openai.error.APIError, openai.error.Timeout, openai.error.APIConnectionError,
                openai.error.ServiceUnavailableError) as e:
            result = None
            trial += 1
            print("OpenAI API error - ", e)
            print("Retry ChatCompletion.")

    return result

def get_encoder_for_model(model: ChatGPTModel | str)->Any:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(model: ChatGPTModel | str, text: str)->int:
    return len(get_encoder_for_model(model).encode(text))

# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def count_token_in_messages(messages: list[dict], model: ChatGPTModel) -> int:
    """Return the number of tokens used by a list of messages."""

    encoding = get_encoder_for_model(model)

    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return count_token_in_messages(messages, model=ChatGPTModel.GPT_3_5_0613)
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return count_token_in_messages(messages, model=ChatGPTModel.GPT_4_0613)
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            try:
                num_tokens += len(encoding.encode(value))
            except:
                print(f"Error on token counting - {key}: {value}")

            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
