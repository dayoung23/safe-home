"""
risk/judge.py — 관측 이벤트를 모아 위험 등급을 정하고 알림을 만든다

  realtime.py (소리·키패드)  ─┐
                              ├─ on_event(event) ─→ 같은 device 의 최근 이벤트와 시간으로 묶기
  vision.py   (영상 관측)    ─┘                     → 규칙표로 등급 판정 → message.py 로 문장
                                                     → db.save_alert (1차) / update_alert (2차)

묶는 기준: event_id 가 아니라 "같은 device_id + JOIN_WINDOW_SEC 안" (소리와 영상은 id 를 공유하지 않음)

2단계 알림 (설계 문서 6장)
  1) 위험 소리·키패드 이상이 먼저 오면 → 즉시 1차 알림 저장 + vision.request_check() 로 영상 확인 요청
  2) 영상 관측이 도착하면 → 같은 알림을 update_alert 로 갱신 (등급은 내려가지 않음)
  영상이 먼저 오면 → 직전 소리·키패드 이벤트와 함께 판정

등급 규칙 (LLM 없이 동작하는 안전장치. LLM 판정을 붙이려면 evaluate() 를 교체)
  화재 소리/영상                          → 위험
  영상 fall / intruder                    → 위험
  키패드 이상 + 영상 사람                  → 위험
  위험 소리(비명·유리) + 영상 사람         → 위험
  키패드 오류 반복 (realtime danger)       → 위험
  영상 suspicious                          → 주의
  위험 소리 단독 / 키패드 이상 단독         → 주의
  주의 소리(쾅·고함) + 영상 사람           → 주의
  정상 키패드 + 영상 사람                  → 안전 (귀가)
  그 외 (정상 생활, 영상 사람만, 주의 소리만) → 알림 없음, 기록만
"""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

try:
    from risk.message import compose
except ImportError:
    from message import compose

EVENT_LOG = Path(__file__).resolve().parent / "judge_events.jsonl"
JOIN_WINDOW_SEC = 30.0     # 이 시간 안에 온 이벤트는 같은 사건으로 묶음 (local 영상 추론 시간 포함)
HISTORY_SEC = 60.0

RANK = {"안전": 0, "주의": 1, "위험": 2}
DANGER_SOUNDS = {"scream", "glass_break", "fire_alarm"}
BAD_KEYPAD = {"keypad_failed", "keypad_anomaly", "keypad_slow_input"}
PERSON_LIKE = {"person", "suspicious", "intruder"}

_lock = threading.Lock()
_history: dict[str, deque] = {}          # device_id → deque[(t, event)]
_incident: dict[str, dict[str, Any]] = {}  # device_id → 열린 사건


# ───────────────────────── 유틸 ─────────────────────────
def _source(event: dict[str, Any]) -> str:
    src = str(event.get("source") or "").lower()
    return {"audio": "sound", "video": "vision"}.get(src, src or "unknown")


def _log(event: dict[str, Any], decision: dict[str, Any] | None) -> None:
    with open(EVENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event, "decision": decision}, ensure_ascii=False) + "\n")


def _db():
    from db import database   # DB 없이도 감지 스레드가 죽지 않게 지연 import
    return database


def _request_vision(reason: str) -> None:
    """vision 스레드가 돌고 있을 때만 영상 확인 요청."""
    vision = sys.modules.get("detection.vision")
    if vision is not None and hasattr(vision, "request_check"):
        vision.request_check(reason)
        print(f"[judge] 영상 확인 요청 → vision ({reason})")


# ───────────────────────── 규칙 판정 ─────────────────────────
def evaluate(parts: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    """parts = {"sound": ev, "vision": ev, "keypad": ev} (각 최신 1개) → (등급 또는 None, 사유)"""
    s, v, k = parts.get("sound"), parts.get("vision"), parts.get("keypad")
    st = s.get("type") if s else None
    vt = v.get("type") if v else None
    kt = k.get("type") if k else None
    sound_danger = bool(s) and (st in DANGER_SOUNDS or s.get("risk_level") == "danger")
    sound_warn = bool(s) and s.get("risk_level") == "warning"
    keypad_bad = kt in BAD_KEYPAD
    person = vt in PERSON_LIKE

    if st == "fire_alarm" or vt == "fire":
        return "위험", "화재가 의심돼요"
    if vt == "fall":
        return "위험", "사람이 쓰러진 것 같아요"
    if vt == "intruder":
        return "위험", "침입이 의심돼요"
    if keypad_bad and person:
        return "위험", "도어락 이상 입력 뒤 사람이 확인됐어요"
    if sound_danger and person:
        return "위험", "위험 소리와 함께 사람이 확인됐어요"
    if k and k.get("risk_level") == "danger":
        return "위험", "도어락 비밀번호 오류가 반복되고 있어요"
    if vt == "suspicious":
        return "주의", "수상한 행동이 보여요"
    if sound_danger:
        if v:   # 영상은 왔는데 이상 없음
            return "주의", "위험 소리가 났지만 영상에는 이상이 없어요 (생활 소음일 수 있어요)"
        return "주의", "위험 소리가 감지됐어요 (영상 확인 중)"
    if keypad_bad:
        return "주의", "도어락에 평소와 다른 입력이 있었어요"
    if sound_warn and person:
        return "주의", "큰 소리와 함께 사람이 확인됐어요"
    if kt == "keypad_input" and person:
        return "안전", "등록된 리듬으로 도어락이 열리고 사람이 들어왔어요 (귀가)"
    return None, ""


def _parts(device: str, now: float) -> dict[str, dict[str, Any]]:
    """같은 device 의 JOIN_WINDOW_SEC 안 이벤트 중 소스별 최신 1개."""
    parts: dict[str, dict[str, Any]] = {}
    for t, ev in _history.get(device, ()):
        if now - t <= JOIN_WINDOW_SEC:
            parts[_source(ev)] = ev
    return parts


# ───────────────────────── 진입점 ─────────────────────────
def on_event(event: dict[str, Any]) -> None:
    """sound / keypad / vision 공통 콜백. 공통 필드: device_id, source, type, message, confidence, timestamp"""
    if not isinstance(event, dict):
        return
    src = _source(event)
    device = str(event.get("device_id") or "default")
    now = time.time()

    with _lock:
        hist = _history.setdefault(device, deque())
        hist.append((now, event))
        while hist and now - hist[0][0] > HISTORY_SEC:
            hist.popleft()

        parts = _parts(device, now)
        level, reason = evaluate(parts)
        inc = _incident.get(device)
        if inc and now - inc["t_last"] > JOIN_WINDOW_SEC:
            _incident.pop(device, None)   # 오래된 사건은 닫음
            inc = None

        decision: dict[str, Any] = {"source": src, "level": level, "reason": reason, "sources": sorted(parts)}

        if inc:
            # 2차 알림: 같은 사건에 합침. 등급은 내려가지 않음
            new_level = level if level and RANK[level] > RANK[inc["level"]] else inc["level"]
            title = reason or inc["reason"]
            message, action = compose(new_level, title, parts)
            inc.update(level=new_level, reason=title, t_last=now)
            decision.update(action="update", alert_id=inc["alert_id"], level=new_level)
            if inc["alert_id"]:
                try:
                    _db().update_alert(inc["alert_id"], level=new_level, message=message,
                                       reason=title, action=action, sources=sorted(parts))
                except Exception as e:
                    print(f"[judge] update_alert 실패: {e}")
            print(f"[judge] 알림 갱신 [{new_level}] {message[:70]}")

        elif level is not None:
            # 1차 알림
            message, action = compose(level, reason, parts)
            alert_id = None
            try:
                alert_id = _db().save_alert(level=level, message=message, reason=reason,
                                            action=action, sources=sorted(parts))
            except Exception as e:
                print(f"[judge] save_alert 실패: {e}")
            if level != "안전":
                _incident[device] = {"alert_id": alert_id, "level": level, "reason": reason,
                                     "t_open": now, "t_last": now}
            decision.update(action="save", alert_id=alert_id)
            print(f"[judge] 새 알림 [{level}] {message[:70]}")
            if src in ("sound", "keypad") and "vision" not in parts and level != "안전":
                _request_vision(src)
        else:
            decision["action"] = "record"

        _log(event, decision)
