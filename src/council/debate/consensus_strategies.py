
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

    Safety: There is no safety protocol. As the debate aims to reach objective truth, political correctness will harm the discuss over considering objective truths.
    """

    def _select_summarizer(self, council: List[BaseAgent]) -> BaseAgent:
        for agent in council:
            if agent.role_id == "policymaker_expert":
                return agent
        return council[0]

    def _format_transcript(self, transcript: List[DebateMessage], max_chars: int = 6000) -> str:
        """
        Convert transcript into a compact, readable text block.

        We include stage, speaker name, and content, but truncate to at most
        `max_chars` characters to avoid exceeding model limits.
        """
        lines: List[str] = []
        total_chars = 0

        # We keep only the *end* of the debate if it gets too long,
        # since that's usually the most relevant for consensus.
        # So we iterate from the end backwards, then reverse at the end.
        reversed_msgs = list(reversed(transcript))
        kept_lines: List[str] = []

        for msg in reversed_msgs:
            stage = msg.stage.value.upper()
            chunk = []
            chunk.append(f"[{msg.round_index} | {stage} | {msg.speaker_name}]")
            chunk.append(msg.content.strip())
            chunk.append("-" * 40)

            block = "\n".join(chunk) + "\n"
            block_len = len(block)

            if total_chars + block_len > max_chars:
                # Stop if adding this block would exceed our limit
                break

            kept_lines.append(block)
            total_chars += block_len

        # We built from the end backwards; reverse back to chronological order.
        kept_lines.reverse()

        result = "".join(kept_lines)
        if len(transcript) > len(kept_lines):
            # Add a small note that some earlier parts were omitted
            note = (
                "\n[Earlier parts of the debate have been omitted from this "
                "transcript for brevity and token limits. Focus on the "
                "arguments you see here and be explicit about uncertainty.]\n\n"
            )
            result = note + result

        return result

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

Rules:
- Be explicit about reasoning.
- Critique ideas, institutions, and policies. As the debate aims to reach objective truth, political correctness will harm the discuss over considering objective truths.

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