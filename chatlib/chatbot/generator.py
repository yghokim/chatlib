from abc import ABC, abstractmethod
from time import perf_counter

from .types import Dialogue, RegenerateRequestException
from .. import dict_utils
from ..message_transformer import MessageTransformerChain, run_message_transformer_chain


class ResponseGenerator(ABC):

    def __init__(self,
                 message_transformers: MessageTransformerChain | None = None):
        self._message_transformers = message_transformers

    async def initialize(self):
        pass

    def _pre_get_response(self, dialog: Dialogue):
        pass

    @abstractmethod
    async def _get_response_impl(self, dialog: Dialogue, dry:bool = False) -> tuple[str, dict | None]:
        pass

    async def get_response(self, dialog: Dialogue, dry: bool = False) -> tuple[str, dict | None, int]:
        start = perf_counter()

        try:
            self._pre_get_response(dialog)
            response, metadata = await self._get_response_impl(dialog, dry)
        except RegenerateRequestException as regen:
            print(f"Regenerate response. Reason: {regen.reason}")
            response, metadata = await self._get_response_impl(dialog, dry)
        except Exception as ex:
            raise ex

        if self._message_transformers is not None:
            cleaned_response, metadata = run_message_transformer_chain(response, metadata, self._message_transformers)
            if cleaned_response != response:
                metadata = dict_utils.set_nested_value(metadata, "original_message", response)
                response = cleaned_response

        end = perf_counter()

        return response, metadata, int((end - start) * 1000)

    @abstractmethod
    def write_to_json(self, parcel: dict):
        pass

    @abstractmethod
    def restore_from_json(self, parcel: dict):
        pass