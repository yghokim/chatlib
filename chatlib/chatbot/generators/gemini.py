from functools import cache
from typing import Any

from jinja2 import Template

from chatlib.chatbot import ChatCompletionResponseGenerator, ChatCompletionParams, TokenLimitExceedHandler
from chatlib.llm.chat_completion_api import ChatCompletionMessage
from chatlib.llm.integration import GeminiAPI


class GeminiResponseGenerator(ChatCompletionResponseGenerator):
    @classmethod
    @cache
    def get_api(cls) -> GeminiAPI:
        return GeminiAPI()

    def __init__(self, base_instruction: str | Template | None = None,
                 instruction_parameters: dict | None = None,
                 initial_user_message: str | list[ChatCompletionMessage] | None = None,
                 chat_completion_params: ChatCompletionParams | None = None,
                 special_tokens: list[tuple[str, str, Any]] | None = None, verbose: bool = False,
                 token_limit_exceed_handler: TokenLimitExceedHandler | None = None, token_limit_tolerance: int = 1024):
        super().__init__(self.get_api(), "gemini-pro", base_instruction, instruction_parameters, initial_user_message,
                         chat_completion_params, None, special_tokens, verbose, token_limit_exceed_handler,
                         token_limit_tolerance)
