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

st.markdown(
    """
    <style>
        /* Global polish */
        .main, .block-container {padding-top: 1rem;}
        .debate-hero {
            background: linear-gradient(120deg, #0d1b2a, #1b263b 50%, #415a77);
            color: #e0e7ff;
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        }
        .pill {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.15);
            margin-right: 0.4rem;
        }
        .timeline-wrapper {
            max-height: 620px;
            overflow-y: auto;
            padding-right: 0.5rem;
        }
        .timeline-card {
            border: 1px solid #e5e7eb;
            border-left: 6px solid #6366f1;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
            background: #0b1221;
            color: #e5e7eb;
            box-shadow: 0 8px 18px rgba(0,0,0,0.25);
        }
        .timeline-card.rebuttal { border-left-color: #f59e0b; }
        .timeline-card.opening { border-left-color: #22c55e; }
        .timeline-meta { font-size: 0.9rem; color: #cbd5e1; }
        .timeline-speaker { font-size: 1.05rem; font-weight: 700; }
        .expert-card {border: 1px solid #1f2937; border-radius: 14px; padding: 0.8rem; background:#0f172a;}
        .expert-card h3 {margin-bottom: 0.2rem; color: #e2e8f0;}
        .expert-card img {border-radius: 10px; max-height: 170px; object-fit: cover;}
        .timeline-chip {display:inline-block; padding:0.15rem 0.45rem; border-radius:999px; font-size:0.78rem; background:#1e293b; border:1px solid #334155; margin-right:0.35rem; color:#cbd5e1;}
        .timeline-card details {margin-top:0.35rem;}
        .timeline-card summary {cursor:pointer; font-weight:600; color:#c7d2fe;}
        .timeline-card .preview {color:#cbd5e1; line-height:1.5;}
        .timeline-card .full {margin-top:0.35rem; white-space:pre-wrap; line-height:1.5; color:#e2e8f0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="debate-hero">
        <div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">
            <span style="font-size:2rem;">🧠</span>
            <div>
                <div class="pill">Live multi-agent council</div>
                <h1 style="margin:0;">Council of Experts – India Debate Lab</h1>
                <p style="margin:0.2rem 0 0;opacity:0.9;">Track openings, rebuttals, and counters at a glance — no more scrolling wall-of-text.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
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


st.markdown("### Debate setup")
setup_left, setup_right = st.columns([3, 2])

with setup_left:
    prompt = st.text_area(
        "Enter the debate prompt / question",
        placeholder=(
            "Example: To what extent did British colonial policy shape "
            "modern India's economic structure and regional inequalities?"
        ),
        height=120,
    )

with setup_right:
    constraints = st.text_area(
        "Optional constraints / scope",
        value="Be historically grounded, avoid slogans, focus on mechanisms and evidence.",
        height=120,
        help="Narrow the lens so experts don't wander: timeframes, evidence types, or red-lines.",
    )

start_button = st.button(
    "🔥 Start Live Debate",
    type="primary",
    use_container_width=True,
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
    cols = st.columns(2)
    for idx, agent in enumerate(council):
        with cols[idx % 2]:
            with st.container():
                st.markdown('<div class="expert-card">', unsafe_allow_html=True)

                img_path = images.get(agent.role_id)
                if img_path and img_path.exists():
                    st.image(str(img_path), use_column_width=True)
                else:
                    st.write("🔲 (image missing)")

                st.markdown(f"<h3>{agent.name}</h3>", unsafe_allow_html=True)
                st.caption(f"Role ID: `{agent.role_id}`")

                # Placeholder where we'll render the live text
                text_placeholders[agent.role_id] = st.empty()

                st.markdown('</div>', unsafe_allow_html=True)

    return text_placeholders


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
    max-height: 480px;
    overflow-y: auto;
">
{buffer}
</div>
        """,
        unsafe_allow_html=True,
    )


def _stage_label(message: DebateMessage, *, total_agents: int) -> str:
    if message.stage == DebateStage.OPENING:
        return f"Opening statement #{message.round_index + 1}"

    opening_count = total_agents
    rebuttal_round = ((message.round_index - opening_count) // total_agents) + 1
    rebuttal_turn = (message.round_index - opening_count) % total_agents + 1
    return f"Rebuttal round {rebuttal_round}, turn {rebuttal_turn}"


def _render_timeline(
    placeholder: st.delta_generator.DeltaGenerator,
    messages: List[DebateMessage],
    *,
    total_agents: int,
    stage_filter: str,
    preview_chars: int = 360,
) -> None:
    cards: List[str] = []
    for msg in messages:
        if stage_filter == "Opening only" and msg.stage != DebateStage.OPENING:
            continue
        if stage_filter == "Rebuttals only" and msg.stage != DebateStage.REBUTTAL:
            continue

        stage_class = "opening" if msg.stage == DebateStage.OPENING else "rebuttal"
        stage_text = _stage_label(msg, total_agents=total_agents)
        chip = "Opening" if msg.stage == DebateStage.OPENING else "Rebuttal"
        preview = msg.content.replace("\n", " ").strip()
        preview = (preview[: preview_chars] + "…") if len(preview) > preview_chars else preview
        cards.append(
            f"""
            <div class="timeline-card {stage_class}">
                <div class="timeline-speaker">{msg.speaker_name}</div>
                <div class="timeline-meta">
                    <span class="timeline-chip">{chip}</span>
                    {stage_text} • Stage: {msg.stage.value.title()}
                </div>
                <details>
                    <summary>Expand to view full response</summary>
                    <div class="preview">{preview}</div>
                    <div class="full">{msg.content}</div>
                </details>
            </div>
            """
        )

    placeholder.markdown(
        """
        <div class="timeline-wrapper">
            {content}
        </div>
        """.format(content="".join(cards) or "<p style='color:#94a3b8;'>Waiting for the first opening move…</p>"),
        unsafe_allow_html=True,
    )


st.markdown("### Debate workspace")
timeline_tab, experts_tab = st.tabs(["🧭 Debate map", "👥 Expert dashboards"])

with timeline_tab:
    st.caption("See every turn with stage labels and counters. Scroll inside the map, not the whole page.")
    stage_filter = st.radio(
        "Timeline filter",
        options=["All stages", "Opening only", "Rebuttals only"],
        index=0,
        horizontal=True,
        help="Filter the map to just openings or rebuttals to cut down visual noise.",
    )
    timeline_placeholder = st.container()
    _render_timeline(
        timeline_placeholder,
        [],
        total_agents=len(council),
        stage_filter=stage_filter,
    )

with experts_tab:
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

    IMPORTANT: With local models (e.g., Ollama) there is no strict token cap,
    so we include the full transcript for rebuttals.
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
        # Include the full transcript for maximum context
        for msg in transcript_messages:
            label = f"{msg.speaker_name} ({msg.stage.value}, #{msg.round_index})"
            content = f"{label}:\n{msg.content}"
            messages.append(ChatMessage(role="assistant", content=content))

    return messages


def run_live_debate(
    topic: DebateTopic,
    num_rebuttal_rounds: int,
    timeline_placeholder: st.delta_generator.DeltaGenerator,
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
    total_agents = len(council)

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
        _render_timeline(
            timeline_placeholder,
            transcript.messages,
            total_agents=total_agents,
            stage_filter=stage_filter,
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
            _render_timeline(
                timeline_placeholder,
                transcript.messages,
                total_agents=total_agents,
                stage_filter=stage_filter,
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
            timeline_placeholder=timeline_placeholder,
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
    _render_timeline(
        timeline_placeholder,
        result.transcript.messages,
        total_agents=len(council),
        stage_filter=stage_filter,
    )
    with consensus_placeholder:
        st.markdown("### 🧾 Council Consensus (last run)")
        if result.consensus:
            st.markdown(result.consensus.text)
        else:
            st.caption("No consensus generated.")