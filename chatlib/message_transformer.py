import re
from abc import ABC, abstractmethod
from functools import reduce
from re import Pattern
from typing import TypeAlias, Callable

from chatlib import dict_utils


class MessageTransformer(Callable[[str, dict | None], tuple[str, dict | None, bool]], ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def _transform(self, message: str, metadata: dict | None) -> tuple[str, dict | None, bool]:
        pass

    def __call__(self, message: str, metadata: dict | None) -> tuple[str, dict | None, bool]:
        transformed_message, metadata, is_transformed = self._transform(message, metadata)
        if is_transformed:
            metadata = dict_utils.set_nested_value(metadata, ["transformers", self.name], True)
        return transformed_message, metadata, is_transformed


MessageTransformerChain: TypeAlias = list[MessageTransformer]


class SpecialTokenExtractionTransformer(MessageTransformer):

    def __init__(self, name: str, token: str | Pattern, onTokenFound: Callable[[str, dict|None], tuple[str, dict | None]] | None = None):
        super().__init__(name)
        self.token = token
        self.onTokenFound = onTokenFound

    def __call__(self, message: str, metadata: dict | None) -> tuple[str, dict | None, bool]:
        cleaned_message = None
        if isinstance(self.token, Pattern):
            matches = self.token.findall(message)
            if len(matches) > 0:
                cleaned_message = self.token.sub("", message)
        elif self.token in message:
            cleaned_message = message.replace(self.token, "")

        if cleaned_message is not None and self.onTokenFound is not None:
            self.onTokenFound(cleaned_message, metadata)

        return cleaned_message or message, metadata, cleaned_message is not None

    @classmethod
    def remove_all_regex(cls, name: str, pattern: str | Pattern)->'SpecialTokenExtractionTransformer':
        return SpecialTokenExtractionTransformer(name, re.compile(pattern), None)

def run_message_transformer_chain(message: str, metadata: dict | None, chain: MessageTransformerChain) -> tuple[str, dict | None]:
    return reduce(lambda t, transformer: transformer(t), chain, (message, metadata))