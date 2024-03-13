from functools import cache
from typing import Callable, Awaitable, Any

from jinja2 import Template

from chatlib.chatbot import ChatCompletionResponseGenerator, TokenLimitExceedHandler, ChatCompletionParams
from chatlib.llm.chat_completion_api import ChatCompletionMessage
from chatlib.llm.integration import AzureLlama2ChatCompletionAPI


class Llama2ResponseGenerator(ChatCompletionResponseGenerator):

    @classmethod
    @cache
    def get_api(cls) -> AzureLlama2ChatCompletionAPI:
        return AzureLlama2ChatCompletionAPI()

    def __init__(self, base_instruction: str | Template | None = None,
                 instruction_parameters: dict | None = None,
                 initial_user_message: str | list[ChatCompletionMessage] | None = None,
                 chat_completion_params: ChatCompletionParams | None = None,
                 function_handler: Callable[[str, dict | None], Awaitable[Any]] | None = None,
                 special_tokens: list[tuple[str, str, Any]] | None = None, verbose: bool = False,
                 token_limit_exceed_handler: TokenLimitExceedHandler | None = None, token_limit_tolerance: int = 1024):
        super().__init__(self.get_api(), "Llama-2-70b-chat-hf", base_instruction, instruction_parameters,
                         initial_user_message,
                         chat_completion_params, function_handler, special_tokens, verbose, token_limit_exceed_handler,
                         token_limit_tolerance)
