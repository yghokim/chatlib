import asyncio

from questionary import prompt

from chatlib.chatbot.generators import ChatGPTResponseGenerator
from chatlib.chatbot.generators.claude import ClaudeResponseGenerator
from chatlib.chatbot.generators.cohere import CohereResponseGenerator
from chatlib.chatbot.generators.gemini import GeminiResponseGenerator
from chatlib.chatbot.generators.llama import Llama2ResponseGenerator
from chatlib.chatbot.generators.together import TogetherAIResponseGenerator
from chatlib.global_config import GlobalConfig
from chatlib.llm.integration import AnthropicModel
from chatlib.llm.integration import ChatGPTModel
from chatlib.llm.integration import TogetherAIModel
from chatlib.utils import cli

agent_args = dict(
    base_instruction="You are a helpful assistant that asks the user about their daily activity and feelings. "
                     "If the user wants to finish the conversation, put special token <|Terminate|> at the end of message.",
    initial_user_message="Hi!",
    special_tokens=[("<|Terminate|>", "terminate", True)]
)

if __name__ == "__main__":
    GlobalConfig.is_cli_mode = True

    answer = prompt([{
        'type': 'select',
        "name": "agent_model",
        'choices': ['GPT4', 'Llama2', 'Gemini', "Mixtral", "Vicuna", "Claude", "Cohere"],
        'default': 'GPT4',
        'message': 'Select model you want to converse with:'
    }])

    agent_model = answer['agent_model']

    if agent_model == 'GPT4':
        agent = ChatGPTResponseGenerator(
            model=ChatGPTModel.GPT_4_latest,
            **agent_args
        )
    elif agent_model == 'Llama2':
        agent = Llama2ResponseGenerator(
            **agent_args
        )
    elif agent_model == 'Gemini':
        agent = GeminiResponseGenerator(
            **agent_args
        )
    elif agent_model == 'Mixtral':
        agent = TogetherAIResponseGenerator(
            model=TogetherAIModel.Mixtral8x7BInstruct,
            **agent_args
        )
    elif agent_model == 'Vicuna':
        agent = TogetherAIResponseGenerator(
            model=TogetherAIModel.Vicuna13B1_5,
            **agent_args
        )
    elif agent_model == 'Claude':
        agent = ClaudeResponseGenerator(model=AnthropicModel.CLAUDE_3_OPUS_20240229, **agent_args)
    elif agent_model == 'Cohere':
        agent = CohereResponseGenerator(**agent_args)
    else:
        raise Exception("Invalid model selected")

    agent.get_api().assert_authorize()

    asyncio.run(cli.run_chat_loop(agent, commands=cli.DEFAULT_TEST_COMMANDS))
