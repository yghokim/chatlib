from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from chatlib.chatbot import ResponseGenerator, Dialogue
from chatlib.chatbot.message_transformer import MessageTransformerChain
from chatlib.utils import dict_utils

StateType = TypeVar('StateType')


class StateBasedResponseGenerator(ResponseGenerator, Generic[StateType], ABC):

    def __init__(self, initial_state: StateType, initial_state_payload: dict | None = None,
                 verbose: bool = False, message_transformers: MessageTransformerChain | None = None):
        super().__init__(message_transformers)
        self.__current_generator: ResponseGenerator | None = None
        self.verbose = verbose

        self.__payload_memory: dict[StateType, dict | None] = dict()

        self.__state_history: list[tuple[StateType, dict | None]] = [(initial_state, initial_state_payload)]

    @property
    def current_state(self) -> StateType:
        return self.__state_history[len(self.__state_history) - 1][0]

    @property
    def current_state_payload(self) -> dict | None:
        return self.__state_history[len(self.__state_history) - 1][1]

    def _push_new_state(self, state: StateType, payload: dict | None):
        self.__state_history.append((state, payload))

    def _get_memoized_payload(self, state: StateType) -> dict | None:
        return self.__payload_memory[state] if state in self.__payload_memory else None

    # Return response generator for a state
    @abstractmethod
    def get_generator(self, state: StateType, payload: dict | None) -> ResponseGenerator:
        pass

    @abstractmethod
    def update_generator(self, generator: ResponseGenerator, payload: dict | None):
        pass

    # Calculate the next state based on the current state and the dialog.
    # Return None if the state does not change.
    @abstractmethod
    async def calc_next_state_info(self, current: StateType, dialog: Dialogue) -> tuple[
                                                                                      StateType | None, dict | None] | None:
        """

        :param current: Current state
        :param dialog: Current dialogue history
        :return:
        1) If None: No state change.
        2) If State not None, the state changes.
        2) If State None but Payload not None => update the current generator with the payload.

        """
        pass

    async def _get_response_impl(self, dialog: Dialogue, dry: bool = False) -> tuple[str, dict | None]:
        if dry is False:  # Update state only when the dry flag is False.
            # Calculate state and update response generator if the state was changed:
            next_state, next_state_payload = await self.calc_next_state_info(self.current_state, dialog) or (None, None)
            if next_state is not None:
                pre_state = self.current_state
                self.__payload_memory[pre_state] = next_state_payload
                self._push_new_state(next_state, next_state_payload)
                self.__current_generator = self.get_generator(self.current_state, self.current_state_payload)
                if self.verbose:
                    print(
                        "▤▤▤▤▤▤▤▤▤▤▤▤ State transition from {} to {} ▤▤▤▤▤▤▤▤▤▤▤▤▤".format(pre_state,
                                                                                           self.current_state))
            elif next_state_payload is not None:  # No state change but generator update.
                print("Update generator with payload.")
                self._push_new_state(self.current_state, next_state_payload)
                self.update_generator(self.__current_generator, next_state_payload)
            elif self.__current_generator is None:  # No state change but initial run.
                self.__current_generator = self.get_generator(self.current_state, self.current_state_payload)

        # Generate response from the child generator:
        message, metadata, elapsed = await self.__current_generator.get_response(dialog, dry)

        metadata = dict_utils.set_nested_value(metadata, "state", self.current_state)
        metadata = dict_utils.set_nested_value(metadata, "payload", self.current_state_payload)

        return message, metadata

    def state_num_appearance(self, state: StateType) -> int:
        """
        Get the number of appearance of the state in the history.
        :param state: state
        :return: number of appearance
        """
        return len([tup for tup in self.__state_history if tup[0] == state])

    @staticmethod
    def trim_dialogue_recent_n_states(dialogue: Dialogue, N: int) -> Dialogue:
        """
        Trim dialogue to only contain turns within recent N turns
        :param dialogue:
        :param N:
        :return:
        """
        accum_num_states = 0
        current_state: StateType = None
        pointer = len(dialogue)
        while pointer > 0:
            pointer -= 1
            if dialogue[pointer].is_user:
                continue
            else:
                state = dialogue[pointer].metadata["state"]
                if current_state != state:
                    if accum_num_states >= N:
                        pointer += 1
                        break
                    else:
                        current_state = state
                        accum_num_states += 1

        return dialogue[pointer:]

    def write_to_json(self, parcel: dict):
        parcel["state_history"] = self.__state_history
        parcel["verbose"] = self.verbose
        parcel["payload_memory"] = self.__payload_memory

    def restore_from_json(self, parcel: dict):
        self.__state_history = parcel["state_history"]
        self.verbose = parcel["verbose"] or False
        self.__payload_memory = parcel["payload_memory"]

        current_state = self.current_state
        pointer = len(self.__state_history) - 1
        while pointer > 0:
            state, payload = self.__state_history[pointer - 1]
            if state != current_state:
                break
            else:
                pointer -= 1

        self.__current_generator = self.get_generator(self.current_state, self.__state_history[pointer][1])
        if pointer < len(self.__state_history) - 1:
            for i in range(pointer + 1, len(self.__state_history)):
                self.update_generator(self.__current_generator, self.__state_history[i][1])
