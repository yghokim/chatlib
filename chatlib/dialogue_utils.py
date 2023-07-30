from typing import Callable

from chatlib.chatbot import Dialogue, DialogueTurn


def find_first_turn(dialogue: Dialogue, test: Callable[[DialogueTurn], bool]) -> tuple[DialogueTurn, int] | None:
    for i in range(len(dialogue)):
        if test(dialogue[i]):
            return dialogue[i], i

    return None


def find_last_turn(dialogue: Dialogue, test: Callable[[DialogueTurn], bool]) -> tuple[DialogueTurn, int] | None:
    for i in range(len(dialogue)-1, -1, -1):
        if test(dialogue[i]):
            return dialogue[i], i

    return None

