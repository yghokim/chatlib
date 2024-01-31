import asyncio

import openai
from chatlib import cli, env_helper
from chatlib.chatbot.generators import ChatGPTResponseGenerator
from chatlib.chatbot.generators.gemini import GeminiResponseGenerator
from chatlib.chatbot.generators.llama import Llama2ResponseGenerator
from chatlib.integration.azure_llama2_utils import AzureLlama2Environment
from chatlib.chatbot import ChatCompletionResponseGenerator
from chatlib.integration.openai_utils import GPTChatCompletionAPI, ChatGPTModel

agent_gpt = ChatGPTResponseGenerator(
        model=ChatGPTModel.GPT_4_latest,
        base_instruction="You are a helpful assistant that asks the user about their daily activity and feelings. "
                         "Put special token <|Terminate|> at the end of message only if the user wants to finish the conversation.",
        initial_user_message="Hi!",
        special_tokens=[("<|Terminate|>", "terminate", True)]
    )

agent_llama = Llama2ResponseGenerator(
        base_instruction="You are a helpful assistant that asks the user about their daily activity and feelings. "
                         "Put special token <|Terminate|> at the end of message only if the user wants to finish the conversation.",
        initial_user_message="Hi!",
        special_tokens=[("<|Terminate|>", "terminate", True)]
    )

agent_gemini = GeminiResponseGenerator(
        base_instruction="You are a helpful assistant that asks the user about their daily activity and feelings. "
                         "Put special token <|Terminate|> at the end of message only if the user wants to finish the conversation.",
        initial_user_message="Hi!",
        special_tokens=[("<|Terminate|>", "terminate", True)]
)


if __name__ == "__main__":

    asyncio.run(cli.run_chat_loop(agent_gemini, commands=cli.DEFAULT_TEST_COMMANDS))
