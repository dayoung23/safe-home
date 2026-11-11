"""
detection/qwen_backup.py — optional Qwen / stub path (legacy in backup/)

Used only when VISION_MODE is qwen|local|stub.
Primary realtime path is YOLO and does not import this at startup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _clip_analyzer():
    try:
        from backup.detection import clip_analyzer as ca
        return ca
    except ImportError as e:
        raise ImportError(
            "Qwen/stub backup 을 쓰려면 backup/detection/clip_analyzer.py 가 필요합니다."
        ) from e


def list_fixtures() -> list[Path]:
    return _clip_analyzer().list_fixtures()


def analyze_clip(
    clip_path: str | None,
    mode: str = "stub",
    fixture: str | Path | None = None,
) -> dict[str, Any]:
    """mode: stub | local (Qwen). 'qwen' is accepted as alias of local."""
    if mode == "qwen":
        mode = "local"
    return _clip_analyzer().analyze_clip(clip_path, mode=mode, fixture=fixture)
