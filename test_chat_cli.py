import asyncio

import openai
from chatlib import cli, env_helper
from chatlib.azure_llama2_utils import AzureLlama2ChatCompletionAPI, AzureLlama2Environment
from chatlib.chatbot import ChatCompletionResponseGenerator
from chatlib.openai_utils import GPTChatCompletionAPI, ChatGPTModel

if __name__ == "__main__":
    # Init OpenAI API
    openai.api_key = env_helper.get_env_variable('OPENAI_API_KEY')
    AzureLlama2Environment.set_host(env_helper.get_env_variable('AZURE_LLAMA2_HOST'))
    AzureLlama2Environment.set_key(env_helper.get_env_variable('AZURE_LLAMA2_KEY'))

    asyncio.run(cli.run_chat_loop(ChatCompletionResponseGenerator(
        api=AzureLlama2ChatCompletionAPI(),
        model=ChatGPTModel.GPT_4_latest,
        base_instruction="You are a helpful assistant that asks the user about their daily activity and feelings. "
                         "Put special token <|Terminate|> at the end of message only if the user wants to finish the conversation.",
        initial_user_message="Hi!",
        special_tokens=[("<|Terminate|>", "terminate", True)]

    ), commands=cli.DEFAULT_TEST_COMMANDS))
