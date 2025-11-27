from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from council.debate.orchestrator import DebateResult
from council.debate.message import DebateMessage


def _project_root() -> Path:
    # council/io/persistence.py -> council/io -> council -> src -> project root
    return Path(__file__).resolve().parents[3]


def _debates_dir() -> Path:
    root = _project_root()
    d = root / "debates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _serialize_message(msg: DebateMessage) -> Dict[str, Any]:
    return {
        "speaker_id": msg.speaker_id,
        "speaker_name": msg.speaker_name,
        "role": msg.role,
        "content": msg.content,
        "stage": msg.stage.value,
        "round_index": msg.round_index,
    }


def save_debate_result(result: DebateResult) -> Path:
    """
    Persist a DebateResult to JSON in the debates/ directory.

    Returns the path of the saved file.
    """
    debates_dir = _debates_dir()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    topic_id = result.transcript.topic.id or "topic"
    filename = f"{ts}_{topic_id}.json"

    data: Dict[str, Any] = {
        "meta": {
            "saved_at_utc": ts,
        },
        "topic": {
            "id": result.transcript.topic.id,
            "title": result.transcript.topic.title,
            "description": result.transcript.topic.description,
            "constraints": result.transcript.topic.constraints,
        },
        "messages": [_serialize_message(m) for m in result.transcript.messages],
        "consensus": (
            {
                "text": result.consensus.text,
                "notes": result.consensus.notes,
            }
            if result.consensus
            else None
        ),
    }

    path = debates_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return path


def list_saved_debates(limit: int = 20) -> List[Path]:
    """
    Return up to `limit` most recent debate JSON files.
    """
    debates_dir = _debates_dir()
    files = sorted(
        debates_dir.glob("*.json"),
        key=lambda p: p.name,
        reverse=True,
    )
    return files[:limit]