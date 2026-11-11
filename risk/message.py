"""
risk/message.py — 알림 문장 만들기 (judge.py 가 등급을 정한 뒤 호출)

판정 로직은 없다. judge 가 넘긴 등급·사유와 각 감지 결과의 message 를 이어 붙여
대시보드 카드에 보일 message / action 문장을 만든다.
(LLM 으로 문장을 다듬고 싶으면 이 파일의 compose() 만 바꾸면 된다)
"""
from __future__ import annotations

from typing import Any

SOURCE_KO = {"sound": "소리", "vision": "영상", "keypad": "도어락"}

ACTION = {
    "위험": "즉시 홈캠을 확인하고, 필요하면 112 또는 119에 신고하세요.",
    "주의": "홈캠 화면으로 상황을 한 번 확인해 보세요.",
    "안전": "",
}


def _vision_text(ev: dict[str, Any]) -> str:
    scene = ev.get("scene") or ""
    msg = ev.get("message") or ""
    return f"{scene}: {msg}" if scene and scene not in msg else msg


def describe(source: str, ev: dict[str, Any]) -> str:
    """감지 결과 1개 → '소리: 비명 소리 감지 - 위험' 같은 한 줄."""
    text = _vision_text(ev) if source == "vision" else (ev.get("message") or "")
    return f"{SOURCE_KO.get(source, source)}: {text}"


def compose(level: str, title: str, parts: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """
    level : 안전 / 주의 / 위험
    title : judge 가 정한 판정 사유 한 문장 (예: "비명과 함께 사람이 확인됐어요")
    parts : {"sound": event, "vision": event, "keypad": event} 중 있는 것만
    반환  : (message, action)
    """
    order = [s for s in ("sound", "keypad", "vision") if s in parts]
    lines = [describe(s, parts[s]) for s in order]
    message = title if not lines else f"{title} — " + " / ".join(lines)
    return message, ACTION.get(level, "")
