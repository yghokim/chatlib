import asyncio

from questionary import prompt

from chatlib.chatbot.generators import ChatGPTResponseGenerator
from chatlib.chatbot.generators.claude import ClaudeResponseGenerator
from chatlib.chatbot.generators.cohere import CohereResponseGenerator
from chatlib.chatbot.generators.gemini import GeminiResponseGenerator
from chatlib.chatbot.generators.llama import Llama2ResponseGenerator
from chatlib.chatbot.generators.together import TogetherAIResponseGenerator
from chatlib.global_config import GlobalConfig
from chatlib.llm.integration import ChatGPTModel
from chatlib.llm.integration import TogetherAIModel
from chatlib.utils import cli

MODELS = ['GPT4', 'Llama2', 'Gemini', "Mixtral", "Vicuna", "Claude", "Cohere"]


#agent_a_args = dict(
#    base_instruction="You are role playing as a police officer who are interrogating a suspect for a burglary. The user acts as a suspect. Try your best to break the user's logic.",
#    initial_user_message="Good evening."
#)

#agent_b_args = dict(
#    base_instruction="You are role playing as a burglar who are interrogated by a police officer. The user acts as a police officer. Try your best to deceive the officer as if you are innocent."
#)


agent_a_args = dict(
    base_instruction="You are role playing as Darth Vader who is revealing that he's a father of Luke Skywalker to Luke. The user acts as Luke.",
    initial_user_message="Good evening."
)

agent_b_args = dict(
    base_instruction="You are role playing as Luke Skywalker who are talking to Darth Vadar. The user acts as Darth Vadar. Try your best to logically refuse that he is your father."
)


if __name__ == "__main__":
    GlobalConfig.is_cli_mode = True

    answer = prompt([{
        'type': 'select',
        "name": "agent_a_model",
        'choices': MODELS,
        'default': 'GPT4',
        'message': 'Select model for Agent A'
    }, {
        'type': 'select',
        "name": "agent_b_model",
        'choices': MODELS,
        'default': 'GPT4',
        'message': 'Select model for Agent B'
    }])

    agent_a_model = answer['agent_a_model']
    agent_b_model = answer['agent_b_model']

    if agent_a_model == 'GPT4':
        agent_a = ChatGPTResponseGenerator(
            model=ChatGPTModel.GPT_4_TURBO,
            **agent_a_args
        )
    elif agent_a_model == 'Llama2':
        agent_a = Llama2ResponseGenerator(
            **agent_a_args
        )
    elif agent_a_model == 'Gemini':
        agent_a = GeminiResponseGenerator(
            **agent_a_args
        )
    elif agent_a_model == 'Mixtral':
        agent_a = TogetherAIResponseGenerator(
            model=TogetherAIModel.Mixtral8x7BInstruct,
            **agent_a_args
        )
    elif agent_a_model == 'Vicuna':
        agent_a = TogetherAIResponseGenerator(
            model=TogetherAIModel.Vicuna13B1_5,
            **agent_a_args
        )
    elif agent_a_model == 'Claude':
        agent_a = ClaudeResponseGenerator(**agent_a_args)
    elif agent_a_model == 'Cohere':
        agent_a = CohereResponseGenerator(**agent_a_args)
    else:
        raise Exception("Invalid model selected")

    agent_a.get_api().assert_authorize()

    if agent_b_model == 'GPT4':
        agent_b = ChatGPTResponseGenerator(
            model=ChatGPTModel.GPT_4_TURBO,
            **agent_b_args
        )
    elif agent_b_model == 'Llama2':
        agent_b = Llama2ResponseGenerator(
            **agent_b_args
        )
    elif agent_b_model == 'Gemini':
        agent_b = GeminiResponseGenerator(
            **agent_b_args
        )
    elif agent_b_model == 'Mixtral':
        agent_b = TogetherAIResponseGenerator(
            model=TogetherAIModel.Mixtral8x7BInstruct,
            **agent_b_args
        )
    elif agent_b_model == 'Vicuna':
        agent_b = TogetherAIResponseGenerator(
            model=TogetherAIModel.Vicuna13B1_5,
            **agent_b_args
        )
    elif agent_b_model == 'Claude':
        agent_b = ClaudeResponseGenerator(**agent_b_args)
    elif agent_b_model == 'Cohere':
        agent_b = CohereResponseGenerator(**agent_b_args)
    else:
        raise Exception("Invalid model selected")

    agent_b.get_api().assert_authorize()

    asyncio.run(cli.run_auto_chat_loop(agent_a, agent_b, user_alias="Luke", ai_alias="Darth Vadar"))
