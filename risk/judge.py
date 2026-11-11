"""
risk/judge.py — 관측 이벤트를 "사건"으로 묶어 위험 등급을 정하고 알림을 만든다

  realtime.py (소리·키패드) ─┐
                             ├─ on_event(event) → 사건 창으로 묶기 → 규칙 판정 → db.save_alert / update_alert
  vision.py   (YOLO 영상)   ─┘

사건 창 (incident window)
  ① 첫 알림거리 이벤트가 오면 사건을 열고 즉시 1차 알림을 저장 (확인 중)
     · 소리·키패드가 먼저면 vision.request_check() 로 지금 프레임을 YOLO 로 바로 확인 요청
  ② FINALIZE_SEC(5초) 동안 들어온 이벤트 + 사건 직전 LOOKBACK_SEC 의 이벤트를 모두 같은 사건으로 묶음
     (그 사이 이벤트가 오면 같은 알림을 바로 갱신)
  ③ 5초가 지나면 묶인 결과로 최종 판정 → 같은 알림을 "종합 판정"으로 갱신
     (LLM 을 붙일 자리: final_judge())
  ④ 최종 판정 뒤에도 JOIN_WINDOW_SEC 안에 오는 이벤트는 같은 알림을 갱신 (등급은 내려가지 않음)

묶는 기준은 event_id 가 아니라 "같은 device_id + 시간" (소리와 영상은 id 를 공유하지 않음)

등급 규칙 (evaluate)
  화재 소리/영상 → 위험 · 영상 fall / intruder → 위험
  키패드 이상 + 영상 사람 → 위험 · 위험 소리(비명·유리) + 영상 사람 → 위험 · 키패드 오류 반복 → 위험
  영상 suspicious → 주의 · 위험 소리 단독 / 키패드 이상 단독 → 주의 · 큰 소리 + 사람 → 주의
  정상 키패드 + 영상 사람 → 안전 (귀가)
  그 외 (사람만 보임, 정상 생활, 큰 소리만) → 알림 없음, 기록만
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
FINALIZE_SEC = 5.0         # 사건 창: 첫 이벤트 후 이만큼 모아서 최종 판정
LOOKBACK_SEC = 10.0        # 사건이 열리기 직전 이만큼의 이벤트도 같은 사건에 포함
JOIN_WINDOW_SEC = 30.0     # 마지막 이벤트 후 이 시간이 지나면 사건 종료
HISTORY_SEC = 60.0

RANK = {"안전": 0, "주의": 1, "위험": 2}
DANGER_SOUNDS = {"scream", "glass_break", "fire_alarm"}
BAD_KEYPAD = {"keypad_failed", "keypad_anomaly", "keypad_slow_input"}
PERSON_LIKE = {"person", "suspicious", "intruder"}

_lock = threading.Lock()
_history: dict[str, deque] = {}             # device_id → deque[(t, event)]
_incident: dict[str, dict[str, Any]] = {}   # device_id → 열린 사건
_seq = 0


# ───────────────────────── 유틸 ─────────────────────────
def _source(event: dict[str, Any]) -> str:
    src = str(event.get("source") or "").lower()
    return {"audio": "sound", "video": "vision"}.get(src, src or "unknown")


def _log(record: dict[str, Any]) -> None:
    try:
        with open(EVENT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _db():
    from db import database   # DB 없이도 감지 스레드가 죽지 않게 지연 import
    return database


def _request_vision(reason: str) -> None:
    """vision 스레드가 돌고 있을 때만 영상 확인 요청 (YOLO 가 다음 프레임을 바로 관측)."""
    vision = sys.modules.get("detection.vision")
    if vision is not None and hasattr(vision, "request_check"):
        vision.request_check(reason)
        print(f"[judge] 영상 확인 요청 → vision ({reason})")


def _save(level, message, reason, action, sources):
    try:
        return _db().save_alert(level=level, message=message, reason=reason, action=action, sources=sources)
    except Exception as e:
        print(f"[judge] save_alert 실패: {e}")
        return None


def _update(alert_id, **fields):
    if not alert_id:
        return
    try:
        _db().update_alert(alert_id, **fields)
    except Exception as e:
        print(f"[judge] update_alert 실패: {e}")


# ───────────────────────── 규칙 판정 ─────────────────────────
def evaluate(parts: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    """parts = {"sound": ev, "vision": ev, "keypad": ev} (소스별 최신 1개) → (등급 또는 None, 사유)"""
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
        return "주의", "수상한 상황이 보여요"
    if sound_danger:
        if v:   # 영상은 확인했는데 사람/이상 없음
            return "주의", "위험 소리가 났지만 영상에는 이상이 없어요 (생활 소음일 수 있어요)"
        return "주의", "위험 소리가 감지됐어요"
    if keypad_bad:
        return "주의", "도어락에 평소와 다른 입력이 있었어요"
    if sound_warn and person:
        return "주의", "큰 소리와 함께 사람이 확인됐어요"
    if kt == "keypad_input" and person:
        return "안전", "등록된 리듬으로 도어락이 열리고 사람이 들어왔어요 (귀가)"
    return None, ""


def final_judge(parts: dict[str, dict[str, Any]], current_level: str) -> tuple[str, str]:
    """사건 창이 끝났을 때의 최종 판정. 지금은 규칙. (LLM 을 붙이면 여기서 호출하고 실패 시 규칙 사용)
    안전장치: 이미 올라간 등급보다 낮추지 않는다."""
    level, reason = evaluate(parts)
    if level is None or RANK[level] < RANK[current_level]:
        return current_level, reason or ""
    return level, reason


def _parts(device: str, since: float) -> dict[str, dict[str, Any]]:
    """같은 device 의 since 이후 이벤트 중 소스별 최신 1개."""
    parts: dict[str, dict[str, Any]] = {}
    for t, ev in _history.get(device, ()):
        if t >= since:
            parts[_source(ev)] = ev
    return parts


# ───────────────────────── 사건 창 최종 판정 ─────────────────────────
def _finalize(device: str, inc_id: int) -> None:
    with _lock:
        inc = _incident.get(device)
        if not inc or inc["id"] != inc_id:
            return
        parts = _parts(device, inc["since"])
        level, reason = final_judge(parts, inc["level"])
        title = f"{reason or inc['reason']} · {int(FINALIZE_SEC)}초 종합 판정"
        message, action = compose(level, title, parts)
        inc.update(level=level, reason=reason or inc["reason"], final=True)
        _update(inc["alert_id"], level=level, message=message, reason=title,
                action=action, sources=sorted(parts))
        _log({"finalize": device, "alert_id": inc["alert_id"], "level": level,
              "reason": title, "sources": sorted(parts)})
    print(f"[judge] 사건 최종 판정 [{level}] {message[:70]}")


# ───────────────────────── 진입점 ─────────────────────────
def on_event(event: dict[str, Any]) -> None:
    """sound / keypad / vision 공통 콜백. 공통 필드: device_id, source, type, message, confidence, timestamp"""
    global _seq
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

        inc = _incident.get(device)
        if inc and now - inc["t_last"] > JOIN_WINDOW_SEC:
            _incident.pop(device, None)   # 오래된 사건은 닫음
            inc = None

        decision: dict[str, Any] = {"source": src, "type": event.get("type")}

        if inc:
            # 열린 사건에 합침 → 같은 알림 갱신 (등급은 내려가지 않음)
            parts = _parts(device, inc["since"])
            level, reason = evaluate(parts)
            new_level = level if level and RANK[level] > RANK[inc["level"]] else inc["level"]
            title = reason or inc["reason"]
            if not inc["final"]:
                title += " (확인 중)"
            message, action = compose(new_level, title, parts)
            inc.update(level=new_level, reason=reason or inc["reason"], t_last=now)
            _update(inc["alert_id"], level=new_level, message=message, reason=title,
                    action=action, sources=sorted(parts))
            decision.update(action="update", alert_id=inc["alert_id"], level=new_level)
            print(f"[judge] 알림 갱신 [{new_level}] {message[:70]}")
        else:
            parts = _parts(device, now - LOOKBACK_SEC)   # 직전 이벤트까지 포함해서 판정
            level, reason = evaluate(parts)
            if level is None:
                decision["action"] = "record"      # 정상 생활 → 기록만
            else:
                title = reason if level == "안전" else f"{reason} (확인 중)"
                message, action = compose(level, title, parts)
                alert_id = _save(level, message, title, action, sorted(parts))
                decision.update(action="save", alert_id=alert_id, level=level)
                print(f"[judge] 새 알림 [{level}] {message[:70]}")
                if level != "안전":                  # 귀가 알림은 사건을 열지 않음
                    _seq += 1
                    _incident[device] = {"id": _seq, "alert_id": alert_id, "level": level,
                                         "reason": reason, "since": now - LOOKBACK_SEC,
                                         "t_open": now, "t_last": now, "final": False}
                    t = threading.Timer(FINALIZE_SEC, _finalize, args=(device, _seq))
                    t.daemon = True
                    t.start()
                    if "vision" not in parts:
                        _request_vision(src)

        _log({"event": event, "decision": decision})
