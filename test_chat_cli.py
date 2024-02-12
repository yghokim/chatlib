import asyncio

from questionary import prompt

from chatlib import cli
from chatlib.chatbot.generators import ChatGPTResponseGenerator
from chatlib.chatbot.generators.claude import ClaudeResponseGenerator
from chatlib.chatbot.generators.gemini import GeminiResponseGenerator
from chatlib.chatbot.generators.llama import Llama2ResponseGenerator
from chatlib.chatbot.generators.together import TogetherAIResponseGenerator
from chatlib.integration.openai_api import ChatGPTModel
from chatlib.integration.together_api import TogetherAIModel
from chatlib.global_config import GlobalConfig


agent_args = dict(
    base_instruction="You are a helpful assistant that asks the user about their daily activity and feelings. "
                     "If the user wants to finish the conversation, put special token <|Terminate|> at the end of message.",
    initial_user_message="Hi!",
    special_tokens=[("<|Terminate|>", "terminate", True)]
)

agent_gpt = ChatGPTResponseGenerator(
        model=ChatGPTModel.GPT_4_latest,
        **agent_args
    )

agent_llama = Llama2ResponseGenerator(
        **agent_args
    )

agent_gemini = GeminiResponseGenerator(
        **agent_args
)


agent_mixtral = TogetherAIResponseGenerator(
        model=TogetherAIModel.Mixtral8x7BInstruct,
        **agent_args
)

agent_vicuna = TogetherAIResponseGenerator(
        model=TogetherAIModel.Vicuna13B1_5,
        **agent_args
)

agent_claude = ClaudeResponseGenerator(**agent_args)


if __name__ == "__main__":
    GlobalConfig.is_cli_mode = True

    answer = prompt([{
        'type': 'select',
        "name": "agent_model",
        'choices': ['GPT', 'Llama2', 'Gemini', "Mixtral", "Vicuna", "Claude"],
        'message': 'Select model you want to converse with:'
    }])

    agent_model = answer['agent_model']

    if agent_model == 'GPT':
        agent = agent_gpt
    elif agent_model == 'Llama2':
        agent = agent_llama
    elif agent_model == 'Gemini':
        agent = agent_gemini
    elif agent_model == 'Mixtral':
        agent = agent_mixtral
    elif agent_model == 'Vicuna':
        agent = agent_vicuna
    elif agent_model == 'Claude':
        agent = agent_claude
    else:
        raise Exception("Invalid model selected")

    agent.get_api().assert_authorize()

    asyncio.run(cli.run_chat_loop(agent, commands=cli.DEFAULT_TEST_COMMANDS))
