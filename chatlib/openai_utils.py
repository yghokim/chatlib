from asyncio import to_thread
from enum import StrEnum
from typing import Any

import openai
import tiktoken

from chatlib.chat_completion import ChatCompletionMessage, ChatCompletionAPI


class ChatGPTModel(StrEnum):
    GPT_3_5_lateste = "gpt-3.5-turbo"
    GPT_3_5_16k_latest = "gpt-3.5-turbo-16k"
    GPT_4_latest = "gpt-4"
    GPT_4_32k_latest = "gpt-4-32k"
    GPT_4_0613 = "gpt-4-0613"
    GPT_3_5_0613 = "gpt-3.5-turbo-0613"


def get_token_limit(model: str):
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


class GPTChatCompletionAPI(ChatCompletionAPI):

    def is_messages_within_token_limit(self, messages: list[ChatCompletionMessage], model: str,
                                       tolerance: int = 120) -> bool:
        return self.count_token_in_messages(messages, model) < get_token_limit(model) - tolerance

    async def run_chat_completion(self, model: str, messages: list[ChatCompletionMessage], params: dict,
                                  trial_count: int = 5) -> Any:
        trial = 0
        result = None
        while trial <= trial_count and result is None:
            try:
                result = await to_thread(openai.ChatCompletion.create,
                                         model=model,
                                         messages=[message.to_dict() for message in messages],
                                         **params
                                         )
            except (openai.error.APIError, openai.error.Timeout, openai.error.APIConnectionError,
                    openai.error.ServiceUnavailableError) as e:
                result = None
                trial += 1
                print("OpenAI API error - ", e)
                print("Retry ChatCompletion.")

        return result

    def count_token_in_messages(self, messages: list[ChatCompletionMessage], model: str) -> int:
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
            return self.count_token_in_messages(messages, model=ChatGPTModel.GPT_3_5_0613)
        elif "gpt-4" in model:
            print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.count_token_in_messages(messages, model=ChatGPTModel.GPT_4_0613)
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.to_dict().items():
                try:
                    num_tokens += len(encoding.encode(value))
                except:
                    print(f"Error on token counting - {key}: {value}")

                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens


def get_encoder_for_model(model: ChatGPTModel | str) -> Any:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        return tiktoken.get_encoding("cl100k_base")
