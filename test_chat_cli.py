import asyncio

from questionary import prompt

from chatlib import cli
from chatlib.chatbot.generators import ChatGPTResponseGenerator
from chatlib.chatbot.generators.gemini import GeminiResponseGenerator
from chatlib.chatbot.generators.llama import Llama2ResponseGenerator
from chatlib.integration.openai_utils import ChatGPTModel
from global_config import GlobalConfig


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
    GlobalConfig.is_cli_mode = True

    answer = prompt([{
        'type': 'select',
        "name": "agent_model",
        'choices': ['GPT', 'Llama2', 'Gemini'],
        'message': 'Select model you want to converse with:'
    }])

    agent_model = answer['agent_model']

    if agent_model == 'GPT':
        agent = agent_gpt
    elif agent_model == 'Llama2':
        agent = agent_llama
    elif agent_model == 'Gemini':
        agent = agent_gemini
    else:
        raise Exception("Invalid model selected")

    agent.get_api().assert_authorize()

    asyncio.run(cli.run_chat_loop(agent_gemini, commands=cli.DEFAULT_TEST_COMMANDS))
