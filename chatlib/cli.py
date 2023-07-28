from os import getcwd, path

from nanoid import generate as generate_id
from yaspin import yaspin
from chatlib.chatbot import ResponseGenerator, TurnTakingChatSession, DialogueTurn, MultiAgentChatSession


CLEAR_LINE = '\033[K'

def __turn_to_string(turn: DialogueTurn) -> str:
    if turn.is_user:
        return f"<User> {turn.message}"
    else:
        return f"<AI> {turn.message} ({turn.metadata.__str__() if turn.metadata is not None else None}) - {turn.processing_time} sec"


async def run_chat_loop(response_generator: ResponseGenerator):
    session_id = generate_id()

    print(f"Start a chat session (id: {session_id}).")
    session = TurnTakingChatSession(session_id, response_generator)

    spinner = yaspin(text="Thinking...")
    spinner.start()
    system_turn = await session.initialize()
    spinner.stop()
    print(__turn_to_string(system_turn))  # Print initial message

    while True:
        user_message = input("You: ")
        spinner.start()
        system_turn = await session.push_user_message(DialogueTurn(user_message, is_user=True))
        spinner.stop()
        print(__turn_to_string(system_turn))


async def run_auto_chat_loop(agent_generator: ResponseGenerator, user_generator: ResponseGenerator,
                             output_path: str = None):
    session_id = generate_id()

    print(f"Start a chat session (id: {session_id}).")
    session = MultiAgentChatSession(session_id, agent_generator, user_generator)
    dialogue = await session.generate_conversation(8, lambda turn: print(__turn_to_string(turn)))

    output_path = output_path or path.join(getcwd(), f"auto_chat_{session_id}.txt")
    with open(output_path, "w", encoding='utf-8') as f:
        f.writelines([f"{__turn_to_string(turn)}\n" for i, turn in enumerate(dialogue)])

    print(f"\nSaved conversation at {output_path}")
