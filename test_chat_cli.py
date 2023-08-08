import asyncio
from os import path, getcwd, getenv

import openai
from dotenv import load_dotenv

from chatlib import cli
from chatlib.chatbot.generators import ChatGPTResponseGenerator

if __name__ == "__main__":
    # Init OpenAI API
    load_dotenv(path.join(getcwd(), ".env"))
    openai.api_key = getenv('OPENAI_API_KEY')

    asyncio.run(cli.run_chat_loop(ChatGPTResponseGenerator(
        base_instruction="You are a helpful assistant that asks the user about their daily activity and feelings. "
                         "Put special token <|Terminate|> at the end of message if the user wants to finish the conversation.",
        initial_user_message="Hi!",
        special_tokens=[("<|Terminate|>", "terminate", True)]

    ), commands=cli.DEFAULT_TEST_COMMANDS))
