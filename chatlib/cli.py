import asyncio
from os import getcwd, path
from typing import Callable, TypeAlias, Awaitable

import questionary

from nanoid import generate as generate_id
from yaspin import yaspin

from chatlib.chat_completion_api import make_non_empty_string_validator
from chatlib.chatbot import ResponseGenerator, TurnTakingChatSession, DialogueTurn, MultiAgentChatSession

CommandDef: TypeAlias = tuple[list[str] | str, Callable[[TurnTakingChatSession], None | Awaitable]]

CLEAR_LINE = '\033[K'


async def regen(session: TurnTakingChatSession):
    system_turn = await session.regenerate_last_system_message()
    if system_turn is not None:
        print(__turn_to_string(system_turn))


DEFAULT_TEST_COMMANDS = [
    ("regen()", regen)
]


def __turn_to_string(turn: DialogueTurn) -> str:
    if turn.is_user:
        return f"<You> {turn.message}"
    else:
        return f"<AI> {turn.message} ({turn.metadata.__str__() if turn.metadata is not None else None}) - {turn.processing_time} sec"


async def run_chat_loop(response_generator: ResponseGenerator, commands: list[CommandDef] | None = None):
    session_id = generate_id()

    print(f"Start a chat session (id: {session_id}).")
    session = TurnTakingChatSession(session_id, response_generator)

    await run_chat_loop_from_session(session, initialize=True, commands=commands)


async def run_chat_loop_from_session(session: TurnTakingChatSession, initialize: bool = False,
                                     commands: list[CommandDef] | None = None):
    print(f"Resume chat for session {session.id}")

    spinner = yaspin(text="Thinking...")

    if initialize:
        spinner.start()
        system_turn = await session.initialize()
        spinner.stop()
        questionary.print(__turn_to_string(system_turn))  # Print initial message
    elif len(session.dialog) > 0:
        for turn in session.dialog:
            questionary.print(__turn_to_string(turn))

        questionary.print("========Continue chat==========")

    while True:
        user_message = await questionary.text("<You>", validate=make_non_empty_string_validator("A message should not be empty.")).ask_async()


        matched_command_actions = [action for cmd, action in commands if (user_message.lower() == cmd if isinstance(cmd,
                                                                                                                    str) else user_message.lower() in cmd)] if commands is not None else []
        if len(matched_command_actions) > 0:
            print("Run command.")
            spinner.start()
            for action in matched_command_actions:
                if asyncio.iscoroutinefunction(action):
                    await action(session)
                else:
                    action(session)
            spinner.stop()
        else:
            spinner.start()
            system_turn = await session.push_user_message(DialogueTurn(user_message, is_user=True))
            spinner.stop()
            questionary.print(__turn_to_string(system_turn))


async def run_auto_chat_loop(agent_generator: ResponseGenerator, user_generator: ResponseGenerator,
                             max_turns: int = 8,
                             output_path: str = None):
    session_id = generate_id()

    print(f"Start a chat session (id: {session_id}).")
    session = MultiAgentChatSession(session_id, agent_generator, user_generator)
    dialogue = await session.generate_conversation(max_turns, lambda turn: print(__turn_to_string(turn)))

    output_path = output_path or path.join(getcwd(), f"auto_chat_{session_id}.txt")
    with open(output_path, "w", encoding='utf-8') as f:
        f.writelines([f"{__turn_to_string(turn)}\n" for i, turn in enumerate(dialogue)])

    print(f"\nSaved conversation at {output_path}")
