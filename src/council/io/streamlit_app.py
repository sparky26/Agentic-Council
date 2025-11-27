from __future__ import annotations
# --- make 'src' importable so 'import council' works ---
import sys
from pathlib import Path

# This file is: .../Agentic council/src/council/io/streamlit_app.py
# We want to add: .../Agentic council/src  to sys.path
SRC_ROOT = Path(__file__).resolve().parents[2]  # points to 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
# --- end of path fix ---

import uuid
from typing import Dict, List

import streamlit as st

MAX_CONTEXT_MESSAGES = 6

from council.agents.council_factory import create_council
from council.agents.base_agent import BaseAgent
from council.config.settings import get_settings
from council.debate.debate_topic import DebateTopic
from council.debate.debate_protocol import BasicDebateProtocol, RoundConfig
from council.debate.message import DebateMessage, DebateStage
from council.debate.consensus_strategies import (
    PolicyLeadConsensusStrategy,
    ConsensusResult,
)
from council.debate.orchestrator import DebateTranscript, DebateResult
from council.llm.base_client import ChatMessage
from council.io.persistence import save_debate_result, list_saved_debates


# ---- Helper: expert images --------------------------------------------------


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _expert_image_dir() -> Path:
    # Expecting "expert images" at project root (unzipped from user's archive)
    return _project_root() / "expert images"


def _role_to_image_path() -> Dict[str, Path]:
    """
    Map agent.role_id to the corresponding expert image path.

    Based on filenames from the attached zip:
    - anthropology expert.png
    - civilizational historian.png
    - indian historian.png
    - policymaker expert.png
    - religion expert.png
    """
    base = _expert_image_dir()
    return {
        "indian_historian": base / "indian historian.png",
        "civilizational_historian": base / "civilizational historian.png",
        "religion_expert": base / "religion expert.png",
        "anthropology_expert": base / "anthropology expert.png",
        "policymaker_expert": base / "policymaker expert.png",
    }


# ---- Streamlit page setup ---------------------------------------------------


st.set_page_config(
    page_title="Council of Experts – India Debate",
    layout="wide",
)

st.title("🧠 Council of Experts – Live Debate on India")

st.markdown(
    """
This interface runs a **live multi-agent debate** on a topic related to India.

- Each expert has their own panel (historian, civilizational historian, religion expert, anthropology expert, policymaker).
- The dialogue streams live under each expert image.
- At the end, the policymaker (by default) synthesizes a **Council Consensus**.
"""
)

# ---- Sidebar: settings & saved debates -------------------------------------


with st.sidebar:
    st.header("Settings")

    num_rebuttal_rounds = st.slider(
        "Number of rebuttal rounds",
        min_value=0,
        max_value=3,
        value=1,
        help="How many full cycles of rebuttals after the opening statements.",
    )

    st.markdown("---")
    st.subheader("Saved debates")

    saved_files = list_saved_debates(limit=10)
    if not saved_files:
        st.caption("No debates saved yet.")
    else:
        for f in saved_files:
            st.caption(f"📄 {f.name}")


# ---- Session state initialization ------------------------------------------


def _init_session_state() -> None:
    if "council" not in st.session_state:
        st.session_state.council: List[BaseAgent] = create_council()
    if "latest_result" not in st.session_state:
        st.session_state.latest_result: DebateResult | None = None
    if "expert_buffers" not in st.session_state:
        # live text per expert
        st.session_state.expert_buffers = {
            agent.role_id: "" for agent in st.session_state.council
        }


_init_session_state()
council: List[BaseAgent] = st.session_state.council


# ---- Debate prompt input ----------------------------------------------------


st.subheader("Debate topic")

prompt = st.text_area(
    "Enter the debate prompt / question",
    placeholder=(
        "Example: To what extent did British colonial policy shape "
        "modern India's economic structure and regional inequalities?"
    ),
    height=100,
)

constraints = st.text_input(
    "Optional constraints / scope (e.g. 'focus on post-1947, avoid moral verdicts')",
    value="Be historically grounded, avoid slogans, focus on mechanisms and evidence.",
)

start_button = st.button(
    "🔥 Start Live Debate",
    type="primary",
    disabled=not bool(prompt.strip()),
)


# ---- Layout for experts -----------------------------------------------------


def build_expert_layout(council: List[BaseAgent]):
    """
    Create a vertical layout: one boxed panel per expert.

    Returns:
    - text_placeholders: dict[role_id -> st.delta_generator.DeltaGenerator]
      Each placeholder will own a streaming text area we update.
    """
    images = _role_to_image_path()
    text_placeholders: Dict[str, st.delta_generator.DeltaGenerator] = {}

    for agent in council:
        # one full-width section per expert
        with st.container(border=True):
            col1, col2 = st.columns([1, 3])

            with col1:
                img_path = images.get(agent.role_id)
                if img_path and img_path.exists():
                    st.image(str(img_path), use_column_width=True)
                else:
                    st.write("🔲 (image missing)")

            with col2:
                st.markdown(f"### {agent.name}")
                st.caption(f"Role ID: `{agent.role_id}`")

                # Placeholder where we'll render the live text
                text_placeholders[agent.role_id] = st.empty()

        # small gap between experts
        st.markdown("---")

    return text_placeholders


st.markdown("### Live Debate")
expert_placeholders = build_expert_layout(council)

status_placeholder = st.empty()
consensus_placeholder = st.container()


# ---- Core live debate logic -------------------------------------------------


def build_topic_from_prompt(prompt: str, constraints: str) -> DebateTopic:
    # Use a random-ish id for persistence
    topic_id = str(uuid.uuid4())[:8]
    title = prompt.strip()[:80] or "Debate Topic"
    return DebateTopic(
        id=topic_id,
        title=title,
        description=prompt.strip(),
        constraints=constraints.strip() or None,
    )


def build_conversation_for_agent(
    *,
    topic: DebateTopic,
    transcript_messages: List[DebateMessage],
    agent: BaseAgent,
    stage: DebateStage,
    rebuttal_round: int | None = None,
) -> List[ChatMessage]:
    """
    Create the ChatMessage sequence that will be sent to a given agent.

    IMPORTANT: To stay within Groq's TPM limits, we only send a *window*
    of recent messages during rebuttal instead of the entire transcript.
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
        """.strip()
    else:
        stage_instructions = f"""
You are {agent.name} participating in a rebuttal round of the council debate.

Task:
- Engage with previous statements from the other experts.
- Point out where you agree and where you disagree, and WHY.
- Bring in additional evidence or reasoning.
- If you revise your earlier position, say so explicitly.
- You may not see the entire transcript; focus on the recent points you see.
        """.strip()

    topic_block = topic.as_user_prompt()
    intro = f"{topic_block}\n\nStage: {stage.value.upper()}\n\n{stage_instructions}"
    if rebuttal_round is not None and stage == DebateStage.REBUTTAL:
        intro += f"\n\nRebuttal round: {rebuttal_round + 1}"

    messages.append(ChatMessage(role="user", content=intro))

    if stage == DebateStage.REBUTTAL and transcript_messages:
        # Only include a sliding window of the most recent messages
        recent = transcript_messages[-MAX_CONTEXT_MESSAGES:]

        for msg in recent:
            label = f"{msg.speaker_name} ({msg.stage.value}, #{msg.round_index})"
            content = f"{label}:\n{msg.content}"
            messages.append(ChatMessage(role="assistant", content=content))

    return messages


def _render_buffer_in_placeholder(placeholder, buffer: str) -> None:
    """Render a scrollable box with the expert's text."""
    placeholder.markdown(
        f"""
<div style="
    white-space: pre-wrap;
    font-family: monospace;
    font-size: 0.9rem;
    border: 1px solid #444;
    padding: 0.5rem;
    border-radius: 0.5rem;
    max-height: 260px;
    overflow-y: auto;
">
{buffer}
</div>
        """,
        unsafe_allow_html=True,
    )


def run_live_debate(
    topic: DebateTopic,
    num_rebuttal_rounds: int,
) -> DebateResult:
    """
    Run a full debate with live streaming into the UI.

    - Uses the existing council from session_state.
    - Streams each agent's turn into their placeholder.
    - At the end, calls the consensus strategy.
    """
    protocol = BasicDebateProtocol(RoundConfig(num_rebuttal_rounds=num_rebuttal_rounds))
    consensus_strategy = PolicyLeadConsensusStrategy()

    transcript = DebateTranscript(topic=topic)
    round_index = 0

    # Opening statements
    status_placeholder.info("Opening statements in progress...")
    for agent in protocol.opening_order(council):
        conv = build_conversation_for_agent(
            topic=topic,
            transcript_messages=transcript.messages,
            agent=agent,
            stage=DebateStage.OPENING,
        )

        placeholder = expert_placeholders.get(agent.role_id)
        role_id = agent.role_id

        if placeholder is None:
            content = agent.respond(conv)
        else:
            buffer = st.session_state.expert_buffers.get(role_id, "")
            for chunk in agent.respond_stream(conv):
                buffer += chunk
                st.session_state.expert_buffers[role_id] = buffer
                _render_buffer_in_placeholder(placeholder, buffer)
            content = buffer or "(no response)"

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
    if num_rebuttal_rounds > 0:
        status_placeholder.info("Rebuttal rounds in progress...")
    for r in range(num_rebuttal_rounds):
        for agent in protocol.rebuttal_order(council):
            conv = build_conversation_for_agent(
                topic=topic,
                transcript_messages=transcript.messages,
                agent=agent,
                stage=DebateStage.REBUTTAL,
                rebuttal_round=r,
            )

            placeholder = expert_placeholders.get(agent.role_id)
            role_id = agent.role_id

            if placeholder is None:
                content = agent.respond(conv)
            else:
                buffer = st.session_state.expert_buffers.get(role_id, "")
                for chunk in agent.respond_stream(conv):
                    buffer += chunk
                    st.session_state.expert_buffers[role_id] = buffer
                    _render_buffer_in_placeholder(placeholder, buffer)
                content = buffer or "(no response)"

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

    status_placeholder.info("Generating council consensus...")
    consensus: ConsensusResult | None = consensus_strategy.generate_consensus(
        topic=topic,
        transcript=transcript.messages,
        council=council,
    )

    result = DebateResult(transcript=transcript, consensus=consensus)
    return result


# ---- Run debate on button click --------------------------------------------


if start_button and prompt.strip():
    topic = build_topic_from_prompt(prompt, constraints)

    with st.spinner("Running live debate..."):
        result = run_live_debate(
            topic=topic,
            num_rebuttal_rounds=num_rebuttal_rounds,
        )

    # Persist in session and to disk
    st.session_state.latest_result = result
    path = save_debate_result(result)
    status_placeholder.success(f"Debate completed and saved to: {path}")

    # Show consensus
    with consensus_placeholder:
        st.markdown("### 🧾 Council Consensus")
        if result.consensus:
            st.markdown(result.consensus.text)
        else:
            st.caption("No consensus generated.")

# If we already have a result from a previous run this session, show its consensus
elif st.session_state.get("latest_result") is not None:
    result: DebateResult = st.session_state.latest_result
    with consensus_placeholder:
        st.markdown("### 🧾 Council Consensus (last run)")
        if result.consensus:
            st.markdown(result.consensus.text)
        else:
            st.caption("No consensus generated.")