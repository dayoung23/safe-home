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
  화재 소리/영상 → 위험 · 영상 fall / intruder → 위험 · 영상 threat(양손 든 위협 자세) → 위험
  키패드 이상 + 영상 사람 → 위험 · 위험 소리(비명·유리) + 영상 사람 → 위험 · 키패드 오류 반복 → 위험
  영상 suspicious → 주의 · 위험 소리 단독 / 키패드 이상 단독 → 주의 · 큰 소리 + 사람 → 주의
  정상 키패드 + 영상 사람 → 안전 (귀가)
  영상 entry(현관 진입) 단독 → ENTRY_WAIT_SEC 동안 키패드 결과를 기다린 뒤에도 없으면 주의
    (키패드 정상 → 귀가 / 키패드 이상 → 위험 으로 위 규칙이 먼저 처리)
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
JOIN_WINDOW_SEC = 15.0     # 마지막 "의미 있는" 이벤트 후 이 시간이 지나면 사건 종료
INCIDENT_MAX_SEC = 30.0    # 사건은 열린 뒤 최대 이 시간까지만 (카메라가 계속 이벤트를 보내도 영원히 안 닫히는 문제 방지)
HISTORY_SEC = 60.0
ENTRY_WAIT_SEC = 8.0       # 현관 진입 후 도어락 인증(키패드) 결과를 기다리는 시간
ENTRY_ALONE = "도어락 인증 없이 현관으로 사람이 들어왔어요"

RANK = {"안전": 0, "주의": 1, "위험": 2}
DANGER_SOUNDS = {"scream", "glass_break", "fire_alarm"}
BAD_KEYPAD = {"keypad_failed", "keypad_anomaly", "keypad_slow_input"}
PERSON_LIKE = {"person", "suspicious", "intruder", "threat", "entry"}

_lock = threading.Lock()
_history: dict[str, deque] = {}             # device_id → deque[(t, event)]
_incident: dict[str, dict[str, Any]] = {}   # device_id → 열린 사건
_seq = 0
_last_save: dict[str, float] = {}           # device_id → 마지막 알림 저장 시각
_recent: deque = deque(maxlen=60)           # 대시보드 디버그용: 최근 이벤트 + 판단 결과
_last_by_source: dict[str, dict[str, Any]] = {}   # 센서 상태 표시용: 소스별 마지막 이벤트


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
    if vt == "threat":
        return "위험", "양손을 든 위협 자세가 계속돼요 (협박 상황 의심)"
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
    if vt == "entry":
        return "주의", ENTRY_ALONE
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
        now = time.time()
        _recent.append({"t": now, "time": time.strftime("%H:%M:%S", time.localtime(now)),
                        "source": "judge", "type": "final", "message": title[:80],
                        "action": "final", "level": level, "demo": False})
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
        if inc and _expired(inc, now):
            _incident.pop(device, None)   # 오래된 사건은 닫음 → 이번 이벤트는 새 사건으로 판단
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
            inc.update(level=new_level, reason=reason or inc["reason"])
            if _notable(src, event):      # 사람만 보임 같은 평범한 이벤트는 사건을 연장하지 않음
                inc["t_last"] = now
            _update(inc["alert_id"], level=new_level, message=message, reason=title,
                    action=action, sources=sorted(parts))
            decision.update(action="update", alert_id=inc["alert_id"], level=new_level)
            print(f"[judge] 알림 갱신 [{new_level}] {message[:70]}")
        else:
            parts = _parts(device, now - LOOKBACK_SEC)   # 직전 이벤트까지 포함해서 판정
            level, reason = evaluate(parts)
            if level is None:
                decision["action"] = "record"      # 정상 생활 → 기록만
            elif reason == ENTRY_ALONE:
                # 현관 진입만 있음 → 바로 침입으로 단정하지 않고 도어락 결과를 잠시 기다림
                t = threading.Timer(ENTRY_WAIT_SEC, _entry_check, args=(device, now))
                t.daemon = True
                t.start()
                decision["action"] = "wait_keypad"
                print(f"[judge] 현관 진입 → 도어락 인증 {int(ENTRY_WAIT_SEC)}초 대기")
            else:
                alert_id = _open(device, level, reason, parts, now)
                decision.update(action="save", alert_id=alert_id, level=level)
                if level != "안전" and "vision" not in parts:
                    _request_vision(src)

        _log({"event": event, "decision": decision})
        _recent.append({
            "t": now, "time": time.strftime("%H:%M:%S", time.localtime(now)),
            "source": src, "type": event.get("type"), "message": str(event.get("message") or "")[:80],
            "action": decision.get("action"), "level": decision.get("level"),
            "demo": bool((event.get("detail") or {}).get("demo")),
        })
        _last_by_source[src] = {"t": now, "type": event.get("type"), "message": str(event.get("message") or "")[:80]}


def _expired(inc: dict[str, Any], now: float) -> bool:
    return now - inc["t_last"] > JOIN_WINDOW_SEC or now - inc["t_open"] > INCIDENT_MAX_SEC


def _notable(src: str, event: dict[str, Any]) -> bool:
    """그 이벤트 하나만으로도 알림거리가 되는가 (소리·도어락 이상, 쓰러짐·위협·현관 진입 등)."""
    level, _ = evaluate({src: event})
    return level is not None


def snapshot(limit: int = 30) -> dict[str, Any]:
    """대시보드용 상태: 센서별 마지막 이벤트(몇 초 전) + 최근 판단 기록 + 열린 사건."""
    now = time.time()
    with _lock:
        sensors = {s: {**v, "ago": round(now - v["t"], 1)} for s, v in _last_by_source.items()}
        recent = list(_recent)[-limit:][::-1]
        incidents = [{"device": d, "level": i["level"], "reason": i["reason"], "final": i["final"],
                      "age": round(now - i["t_open"], 1)}
                     for d, i in _incident.items() if not _expired(i, now)]
    return {"sensors": sensors, "recent": recent, "incidents": incidents,
            "window_sec": FINALIZE_SEC}


def reset(clear_log: bool = True) -> None:
    """시연용: 기억하고 있는 이벤트·사건을 비움 (clear_log=False 면 화면용 판단 로그는 남김)."""
    with _lock:
        _history.clear()
        _incident.clear()
        _last_save.clear()
        if clear_log:
            _recent.clear()
            _last_by_source.clear()


def _open(device: str, level: str, reason: str, parts: dict, now: float):
    """새 알림 저장 (+ 안전이 아니면 사건 창 열기). _lock 을 잡은 상태에서 호출."""
    global _seq
    title = reason if level == "안전" else f"{reason} (확인 중)"
    message, action = compose(level, title, parts)
    alert_id = _save(level, message, title, action, sorted(parts))
    _last_save[device] = now
    print(f"[judge] 새 알림 [{level}] {message[:70]}")
    if level != "안전":                  # 귀가 알림은 사건을 열지 않음
        _seq += 1
        _incident[device] = {"id": _seq, "alert_id": alert_id, "level": level,
                             "reason": reason, "since": now - LOOKBACK_SEC,
                             "t_open": now, "t_last": now, "final": False}
        t = threading.Timer(FINALIZE_SEC, _finalize, args=(device, _seq))
        t.daemon = True
        t.start()
    return alert_id


def _entry_check(device: str, t_entry: float) -> None:
    """현관 진입 ENTRY_WAIT_SEC 후: 그 사이 다른 판정(귀가·도어락 이상 등)이 없었으면 주의 알림."""
    with _lock:
        now = time.time()
        inc = _incident.get(device)
        if (inc and not _expired(inc, now)) or _last_save.get(device, 0) >= t_entry:
            return                       # 이미 다른 이벤트로 처리됨
        parts = _parts(device, t_entry - LOOKBACK_SEC)
        level, reason = evaluate(parts)
        if level is None:
            return
        _open(device, level, reason, parts, now)
        _log({"entry_check": device, "level": level, "reason": reason, "sources": sorted(parts)})
