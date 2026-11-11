"""
RiskJudge 진입점 (팀 확장용 스켈레톤).

vision/sound 실시간 서버는 관측 이벤트를 on_event() 로 넘긴다.
여기서 오디오·비디오 partial 을 event_id 기준으로 모으고,
최종 등급·자연어 알림을 만드는 로직을 팀원이 채운다.

현재: 이벤트 수신·로그·(가능하면) 간단 알림 저장까지.
MongoDB 가 없어도 vision 시연이 깨지지 않도록 DB 는 지연 import.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

EVENT_LOG = Path(__file__).resolve().parent / "judge_events.jsonl"

# event_id → partial bag  (순서 무관 join 용)
_pending: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_TTL_SEC = 15.0


def _append_log(event: dict[str, Any]) -> None:
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _try_save_alert(level: str, message: str, sources: list[str], reason: str = "") -> str | None:
    try:
        from db import database

        return database.save_alert(
            level=level,
            message=message,
            reason=reason,
            action="",
            sources=sources,
        )
    except Exception as e:
        print(f"[judge] alert 저장 스킵: {e}")
        return None


def _source_key(event: dict[str, Any]) -> str:
    src = str(event.get("source") or "").lower()
    if src in ("video", "vision"):
        return "vision"
    if src in ("audio", "sound", "keypad"):
        return "sound"
    return src or "unknown"


def on_event(event: dict[str, Any]) -> None:
    """
    sound / vision 공통 콜백.

    기대 필드(최소):
      source, message, timestamp
      선택: device_id, event_id, confidence, detail, risk_level, ...
    """
    if not isinstance(event, dict):
        print(f"[judge] ignore non-dict event: {type(event)}")
        return

    _append_log(event)
    src = _source_key(event)
    event_id = event.get("event_id") or event.get("clip_id") or event.get("device_id") or "orphan"
    message = event.get("message") or ""
    print(f"[judge] on_event source={src} event_id={event_id} msg={str(message)[:80]}")

    with _lock:
        bag = _pending.setdefault(event_id, {"parts": {}, "created": event.get("timestamp")})
        bag["parts"][src] = event
        parts = dict(bag["parts"])

    # 해커톤 최소 동작:
    # - vision 단독 / sound 단독 이벤트도 1차 알림으로 남길 수 있게 함
    # - 둘 다 모이면 sources 를 합쳐 한 번 더 남김 (본판 LLM 판정은 팀원이 교체)
    if "vision" in parts and "sound" in parts:
        merged_msg = (
            f"[결합] 영상: {parts['vision'].get('message', '')} / "
            f"소리: {parts['sound'].get('message', '')}"
        )
        level = parts["sound"].get("risk_level") or "주의"
        # audio 쪽 risk_level 이 normal|warning|danger 일 수 있음 → 한글 등급 러프 매핑
        level_map = {
            "normal": "안전",
            "warning": "주의",
            "danger": "위험",
            "안전": "안전",
            "주의": "주의",
            "위험": "위험",
        }
        level = level_map.get(str(level), "주의")
        aid = _try_save_alert(level, merged_msg, ["sound", "vision"], reason="join_ready")
        print(f"[judge] join ready event_id={event_id} alert_id={aid}")
        with _lock:
            _pending.pop(event_id, None)
        return

    # 단독 관측: vision 은 관측만, 등급은 보수적으로 주의/안전
    if src == "vision":
        _try_save_alert("주의", str(message), ["vision"], reason="vision_only")
    elif src == "sound":
        rl = str(event.get("risk_level") or "warning")
        level_map = {"normal": "안전", "warning": "주의", "danger": "위험"}
        _try_save_alert(level_map.get(rl, "주의"), str(message), ["sound"], reason="sound_only")
