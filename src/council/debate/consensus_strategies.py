
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from council.agents.base_agent import BaseAgent
from council.debate.debate_topic import DebateTopic
from council.debate.message import DebateMessage, DebateStage
from council.llm.base_client import ChatMessage


@dataclass
class ConsensusResult:
    """
    Result of applying a consensus strategy.

    - text: the synthesized conclusion
    - notes: optional metadata or caveats
    """
    text: str
    notes: Optional[str] = None


class ConsensusStrategy(ABC):
    @abstractmethod
    def generate_consensus(
        self,
        topic: DebateTopic,
        transcript: List[DebateMessage],
        council: List[BaseAgent],
    ) -> ConsensusResult:
        raise NotImplementedError


class PolicyLeadConsensusStrategy(ConsensusStrategy):
    """
    Default consensus strategy:

    - Pick the policymaker agent if present, otherwise the first agent.
    - Provide them with a compact textual representation of the transcript.
    - Ask them to synthesize a grounded, evidence-based "council conclusion"
      highlighting:
        - consensus points
        - major disagreements
        - policy-relevant implications

    The summarizer must weight each expert's contribution by evidentiary
    strength and relevance to the topic rather than by tone or perceived
    politeness.
    """

    def _select_summarizer(self, council: List[BaseAgent]) -> BaseAgent:
        for agent in council:
            if agent.role_id == "policymaker_expert":
                return agent
        return council[0]

    def _format_transcript(self, transcript: List[DebateMessage]) -> str:
        """
        Convert the full transcript into a readable text block.

        With local models and relaxed token limits, we no longer truncate.
        """
        lines: List[str] = []
        for msg in transcript:
            stage = msg.stage.value.upper()
            lines.append(f"[{msg.round_index} | {stage} | {msg.speaker_name}]")
            lines.append(msg.content.strip())
            lines.append("-" * 40)
        return "\n".join(lines)

    def generate_consensus(
        self,
        topic: DebateTopic,
        transcript: List[DebateMessage],
        council: List[BaseAgent],
    ) -> ConsensusResult:
        if not council:
            return ConsensusResult(
                text="No council agents were available to form a consensus.",
                notes="empty_council",
            )

        summarizer = self._select_summarizer(council)
        transcript_text = self._format_transcript(transcript)

        instructions = f"""
You are now acting as the COUNCIL'S CONSENSUS DRAFTER.

Debate topic:
{topic.title}

Description:
{topic.description}

Constraints:
{topic.constraints or "None specified."}

Below is the full transcript of the council's debate, including opening
statements and rebuttals. Your task:

1. Extract the factual points that are well-supported or broadly accepted.
2. Identify key disagreements and why the experts differ.
3. Propose a best-effort, evidence-grounded conclusion that:
   - does not simply average opinions,
   - states where the evidence strongly points,
   - is clear about remaining uncertainty.
4. Translate implications into concrete, realistic considerations for
   contemporary Indian policy or public discourse where relevant.
5. Ensure all five expert perspectives are represented in proportion to the
   strength and relevance of their evidence-backed arguments.

Rules:
- Be explicit about reasoning.
- Critique ideas, institutions, and policies. As the debate aims to reach
  objective truth, avoid political correctness or euphemisms; favor clear,
  topic-anchored facts and trade-offs.
- If an argument is off-topic or weakly supported, down-weight it explicitly;
  if it is well-supported, highlight why and by whom it was offered.

Transcript:
{transcript_text}

Now write the COUNCIL CONSENSUS in the following structure:

1. Core factual points
2. Key disagreements
3. Provisional conclusion
4. Policy / practical implications (if any)
""".strip()

        conversation = [ChatMessage(role="user", content=instructions)]
        consensus_text = summarizer.respond(conversation)

        return ConsensusResult(text=consensus_text, notes=f"summarizer={summarizer.role_id}")
