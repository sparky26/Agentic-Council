
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from council.agents.base_agent import BaseAgent


@dataclass(frozen=True)
class RoundConfig:
    """
    Configuration parameters for the debate.

    - num_rebuttal_rounds: how many full council rebuttal cycles
    """
    num_rebuttal_rounds: int = 1


class DebateProtocol(ABC):
    """
    Abstract protocol defining how a debate proceeds across stages/rounds.

    The orchestrator delegates turn order and number of rounds to this object.
    """

    @abstractmethod
    def opening_order(self, council: List[BaseAgent]) -> List[BaseAgent]:
        """Order in which agents give their opening statements."""
        raise NotImplementedError

    @abstractmethod
    def rebuttal_order(self, council: List[BaseAgent]) -> List[BaseAgent]:
        """Order in which agents give rebuttals."""
        raise NotImplementedError

    @abstractmethod
    def num_rebuttal_rounds(self) -> int:
        """How many rebuttal rounds to run."""
        raise NotImplementedError


class BasicDebateProtocol(DebateProtocol):
    """
    Minimal protocol:

    - Opening: each agent once, in council order
    - Rebuttal: each agent once per round, in council order
    """

    def __init__(self, config: RoundConfig | None = None) -> None:
        self._config = config or RoundConfig()

    def opening_order(self, council: List[BaseAgent]) -> List[BaseAgent]:
        return list(council)

    def rebuttal_order(self, council: List[BaseAgent]) -> List[BaseAgent]:
        return list(council)

    def num_rebuttal_rounds(self) -> int:
        return self._config.num_rebuttal_rounds