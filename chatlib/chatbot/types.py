from typing import TypeAlias, Optional
from pydantic import BaseModel, Field, ConfigDict

import nanoid

from chatlib.time import get_timestamp


class RegenerateRequestException(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class DialogueTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str
    is_user: bool = True
    id: str = Field(default_factory=lambda: nanoid.generate(size=20))
    timestamp: int = Field(default_factory=get_timestamp)
    processing_time: Optional[int] = None
    metadata: dict | None = None


Dialogue: TypeAlias = list[DialogueTurn]
