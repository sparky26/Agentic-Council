from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from council.debate.message import DebateMessage
from council.debate.debate_topic import DebateTopic


@dataclass
class TurnEvaluation:
    """
    Evaluation of a single DebateMessage.

    Keep it simple for now: just some numeric scores and a short note.
    """
    message_index: int
    logical_clarity: float  # 0.0 - 1.0
    use_of_evidence: float  # 0.0 - 1.0
    fairness_to_other_views: float  # 0.0 - 1.0
    notes: Optional[str] = None


class DebateEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        topic: DebateTopic,
        transcript: List[DebateMessage],
    ) -> List[TurnEvaluation]:
        raise NotImplementedError


class NoOpEvaluator(DebateEvaluator):
    """
    Placeholder evaluator that assigns neutral scores.

    You can later plug in an LLM-based or rule-based evaluator instead.
    """

    def evaluate(
        self,
        topic: DebateTopic,
        transcript: List[DebateMessage],
    ) -> List[TurnEvaluation]:
        results: List[TurnEvaluation] = []
        for idx, _msg in enumerate(transcript):
            results.append(
                TurnEvaluation(
                    message_index=idx,
                    logical_clarity=0.5,
                    use_of_evidence=0.5,
                    fairness_to_other_views=0.5,
                    notes="No-op evaluator (placeholder).",
                )
            )
        return results