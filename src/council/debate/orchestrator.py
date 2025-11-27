from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from council.agents.base_agent import BaseAgent
from council.debate.debate_protocol import DebateProtocol
from council.debate.debate_topic import DebateTopic
from council.debate.message import DebateMessage, DebateStage
from council.debate.consensus_strategies import (
    ConsensusStrategy,
    PolicyLeadConsensusStrategy,
    ConsensusResult,
)
from council.llm.base_client import ChatMessage


@dataclass
class DebateTranscript:
    """
    Full record of a debate.

    - topic: the debated topic
    - messages: ordered list of DebateMessage
    """
    topic: DebateTopic
    messages: List[DebateMessage] = field(default_factory=list)


@dataclass
class DebateResult:
    """
    Combined output of a debate run.

    - transcript: all messages
    - consensus: optional synthesized conclusion
    """
    transcript: DebateTranscript
    consensus: Optional[ConsensusResult]


class DebateOrchestrator:
    """
    Coordinates a debate between a council of agents.

    Responsibilities:
    - enforce the protocol (opening + rebuttal rounds)
    - construct the right conversation history for each agent turn
    - delegate to an optional ConsensusStrategy for final synthesis

    It does NOT:
    - know any agent-specific prompts (that's in config + agents)
    - call the LLM directly; it only works through BaseAgent.respond()
    """

    def __init__(
        self,
        protocol: DebateProtocol,
        consensus_strategy: Optional[ConsensusStrategy] = None,
    ) -> None:
        self._protocol = protocol
        self._consensus_strategy = consensus_strategy or PolicyLeadConsensusStrategy()

    # ---- Public API ---------------------------------------------------------

    def run_debate(
        self,
        topic: DebateTopic,
        council: List[BaseAgent],
    ) -> DebateResult:
        transcript = DebateTranscript(topic=topic)
        round_index = 0

        # Opening statements
        for agent in self._protocol.opening_order(council):
            content = self._run_agent_turn(
                topic=topic,
                transcript=transcript,
                agent=agent,
                stage=DebateStage.OPENING,
            )
            transcript.messages.append(
                DebateMessage(
                    speaker_id=agent.role_id,
                    speaker_name=agent.name,
                    role="assistant",
                    content=content,
                    stage=DebateStage.OPENING,
                    round_index=round_index,
                )
            )
            round_index += 1

        # Rebuttal rounds
        for r in range(self._protocol.num_rebuttal_rounds()):
            for agent in self._protocol.rebuttal_order(council):
                content = self._run_agent_turn(
                    topic=topic,
                    transcript=transcript,
                    agent=agent,
                    stage=DebateStage.REBUTTAL,
                    rebuttal_round=r,
                )
                transcript.messages.append(
                    DebateMessage(
                        speaker_id=agent.role_id,
                        speaker_name=agent.name,
                        role="assistant",
                        content=content,
                        stage=DebateStage.REBUTTAL,
                        round_index=round_index,
                    )
                )
                round_index += 1

        # Consensus
        consensus: Optional[ConsensusResult] = None
        if self._consensus_strategy is not None:
            consensus = self._consensus_strategy.generate_consensus(
                topic=topic,
                transcript=transcript.messages,
                council=council,
            )

        return DebateResult(transcript=transcript, consensus=consensus)

    # ---- Internal helpers ---------------------------------------------------

    def _run_agent_turn(
        self,
        *,
        topic: DebateTopic,
        transcript: DebateTranscript,
        agent: BaseAgent,
        stage: DebateStage,
        rebuttal_round: int | None = None,
    ) -> str:
        """
        Build the conversation for a single agent's turn and get their response.
        """
        conversation = self._build_conversation_for_agent(
            topic=topic,
            transcript=transcript,
            agent=agent,
            stage=stage,
            rebuttal_round=rebuttal_round,
        )

        # We just use non-streaming for orchestrated debates.
        return agent.respond(conversation)

    def _build_conversation_for_agent(
        self,
        *,
        topic: DebateTopic,
        transcript: DebateTranscript,
        agent: BaseAgent,
        stage: DebateStage,
        rebuttal_round: int | None = None,
    ) -> List[ChatMessage]:
        """
        Create the ChatMessage sequence that will be sent to a given agent.

        Structure:

        1. A user message with the topic and explicit instructions for the stage.
        2. For opening stage:
           - no prior council messages (agent lays out its initial view).
        3. For rebuttal:
           - include a summary of what has happened so far as user+assistant
             messages, so the agent can respond to others.
        """
        messages: List[ChatMessage] = []

        if stage == DebateStage.OPENING:
            stage_instructions = f"""
You are {agent.name} participating in the opening round of the council debate.

Task:
- Present your analysis of the topic from your expertise.
- Anticipate possible objections from other specialists.
- Be explicit about sources, periods, and uncertainties.
- You are NOT trying to be "balanced" for its own sake; you are trying
  to be accurate, rigorous, and honest about trade-offs.
- Keep every point tightly linked to the stated topic; avoid tangents or
  virtue-signaling and let evidence drive your stance.
            """.strip()
        else:
            stage_instructions = f"""
You are {agent.name} participating in a rebuttal round of the council debate.

Task:
- Engage with previous statements from the other experts.
- Point out where you agree and where you disagree, and WHY.
- Bring in additional evidence or reasoning.
- If you revise your earlier position, say so explicitly.
- Critique arguments based on evidentiary strength and topic relevance,
  and call out any detours into politeness or unrelated issues.
            """.strip()

        topic_block = topic.as_user_prompt()

        intro = f"{topic_block}\n\nStage: {stage.value.upper()}\n\n{stage_instructions}"
        if rebuttal_round is not None and stage == DebateStage.REBUTTAL:
            intro += f"\n\nRebuttal round: {rebuttal_round + 1}"

        messages.append(ChatMessage(role="user", content=intro))

        if stage == DebateStage.REBUTTAL:
            # Add previous council messages as context. We represent each
            # as an assistant message with speaker labels.
            for msg in transcript.messages:
                label = f"{msg.speaker_name} ({msg.stage.value}, #{msg.round_index})"
                content = f"{label}:\n{msg.content}"
                messages.append(ChatMessage(role="assistant", content=content))

        return messages
