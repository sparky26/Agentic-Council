from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DebateTopic:
    """
    Represents a debate topic and its framing.

    - title: short human-readable title
    - description: more detailed explanation of the question
    - constraints: optional guardrails (e.g. "focus on post-1947" or
      "assume perfect information about economic data").
    """
    id: str
    title: str
    description: str
    constraints: Optional[str] = None

    def as_user_prompt(self) -> str:
        """
        Serialize the topic into a single user-message string for LLMs.
        """
        parts = [
            f"Debate topic: {self.title}",
            "",
            f"Description:\n{self.description}",
        ]
        if self.constraints:
            parts.append("")
            parts.append(f"Constraints / scope:\n{self.constraints}")
        return "\n".join(parts)