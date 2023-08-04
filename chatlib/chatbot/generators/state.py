from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from chatlib.chatbot import ResponseGenerator, Dialogue

StateType = TypeVar('StateType')


class StateBasedResponseGenerator(ResponseGenerator, Generic[StateType], ABC):

    def __init__(self,
                 initial_state: StateType,
                 initial_state_payload: dict | None = None,
                 verbose: bool = False
                 ):
        self.__current_generator: ResponseGenerator | None = None
        self.verbose = verbose

        self.__payload_memory: dict[StateType, dict | None] = dict()

        self.__state_history: list[tuple[StateType, list[dict | None]]] = [(initial_state, [initial_state_payload])]

    @property
    def __current_state(self) -> StateType:
        return self.__state_history[len(self.__state_history) - 1][0]

    @property
    def __current_state_payload(self) -> dict | None:
        return self.__state_history[len(self.__state_history) - 1][1][-1]

    def _push_new_state(self, state: StateType, payload: dict | None):
        if self.__state_history[-1][0] == state:
            self.__state_history[-1][1].append(payload)
        else:
            self.__state_history.append((state, [payload]))

    def _get_memoized_payload(self, state: StateType) -> dict | None:
        return self.__payload_memory[state] if state in self.__payload_memory else None

    # Return response generator for a state
    @abstractmethod
    async def get_generator(self, state: StateType, payload: dict | None) -> ResponseGenerator:
        pass

    # Calculate the next state based on the current state and the dialog.
    # Return None if the state does not change.
    @abstractmethod
    async def calc_next_state_info(self, current: StateType, dialog: Dialogue) -> tuple[StateType, dict | None] | None:
        pass

    async def _get_response_impl(self, dialog: Dialogue) -> tuple[str, dict | None]:

        # Calculate state and update response generator if the state was changed:
        next_state, next_state_payload = await self.calc_next_state_info(self.__current_state, dialog) or (None, None)
        if next_state is not None and self.__current_state_payload != next_state_payload:
            pre_state = self.__current_state
            self.__payload_memory[pre_state] = next_state_payload
            self._push_new_state(next_state, next_state_payload)
            self.__current_generator = await self.get_generator(self.__current_state, self.__current_state_payload)
            if self.verbose:
                print(
                    "▤▤▤▤▤▤▤▤▤▤▤▤ State transition from {} to {} ▤▤▤▤▤▤▤▤▤▤▤▤▤".format(pre_state, self.__current_state))
        elif self.__current_generator is None:
            self.__current_generator = await self.get_generator(self.__current_state, self.__current_state_payload)

        # Generate response from the child generator:
        message, metadata = await self.__current_generator._get_response_impl(dialog)

        additional_metadata = {"state": self.__current_state, "payload": self.__current_state_payload}

        if metadata is not None:
            metadata.update(additional_metadata)
        else:
            metadata = additional_metadata

        return message, metadata

    def state_num_appearance(self, state: StateType)->int:
        """
        Get the number of appearance of the state in the history.
        :param state: state
        :return: number of appearance
        """
        return len([tup for tup in self.__state_history if tup[0] == state])

    @staticmethod
    def trim_dialogue_recent_n_states(dialogue: Dialogue, N: int)->Dialogue:
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
        self.__current_generator = None
        self.__payload_memory = parcel["payload_memory"]
