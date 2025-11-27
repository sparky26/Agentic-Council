from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class DebateStage(str, Enum):
    OPENING = "opening"
    REBUTTAL = "rebuttal"
    CONSENSUS = "consensus"


RoleType = Literal["user", "assistant", "system"]


@dataclass
class DebateMessage:
    """
    A single message in the debate transcript.

    - speaker_id: stable id for the source (e.g. "indian_historian", "user")
    - speaker_name: human-friendly label ("Indian Historian", "User")
    - role: chat role from the LLM perspective ("user"/"assistant"/"system")
    - content: the actual text
    - stage: which phase of the debate this belongs to
    - round_index: monotonically increasing integer across the entire debate
    """
    speaker_id: str
    speaker_name: str
    role: RoleType
    content: str
    stage: DebateStage
    round_index: int