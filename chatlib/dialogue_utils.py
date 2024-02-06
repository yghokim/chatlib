from typing import Callable

from chatlib.chatbot import Dialogue, DialogueTurn


def find_first_turn(dialogue: Dialogue, test: Callable[[DialogueTurn], bool]) -> tuple[DialogueTurn | None, int]:
    for i in range(len(dialogue)):
        if test(dialogue[i]):
            return dialogue[i], i

    return None, -1


def find_last_turn(dialogue: Dialogue, test: Callable[[DialogueTurn], bool]) -> tuple[DialogueTurn | None, int]:
    for i in range(len(dialogue) - 1, -1, -1):
        if test(dialogue[i]):
            return dialogue[i], i

    return None, -1


def extract_last_turn_sequence(dialogue: Dialogue, test: Callable[[DialogueTurn], bool]) -> Dialogue:
    seq = []
    for turn in reversed(dialogue):
        if test(turn):
            seq.append(turn)
        else:
            break

    return seq
