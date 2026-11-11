"""
server/routes/demo.py — 대시보드 상태 조회 + 시연 컨트롤 (로그인 필요)

  GET  /api/status        센서별 마지막 감지, 최근 판단 기록, 열린 사건, 카메라 이미지 상태
  POST /api/demo/event    {"kind": "scream"}         가짜 감지 이벤트 1개를 judge 에 넣음
  POST /api/demo/scenario {"name": "intrusion"}      시나리오(이벤트 여러 개를 시간차로) 실행
  POST /api/demo/clear                               알림·judge 기억 초기화

시연 이벤트는 실제 센서 이벤트와 같은 형식으로 judge.on_event 에 들어간다 (detail.demo = True).
판단·알림 저장은 평소와 똑같이 judge 가 한다. DEMO_MODE=0 이면 /api/demo/* 를 막는다.
"""
from __future__ import annotations

import datetime as dt
import os
import threading
import time
import uuid
from pathlib import Path

from flask import Blueprint, jsonify, request, session

from db import database
from server.routes.auth import login_required

bp = Blueprint("demo", __name__, url_prefix="/api")

DEVICE_ID = os.getenv("DEVICE_ID", "homecam-1")
DEMO_ENABLED = os.getenv("DEMO_MODE", "1").strip() not in ("0", "false", "False")
CAMERA_DIR = Path(__file__).resolve().parents[1] / "static" / "camera"

# 실제 센서와 같은 형식의 이벤트 템플릿
EVENTS = {
    "scream":     ("sound", "scream", "danger", "비명 소리 감지 - 위험"),
    "glass":      ("sound", "glass_break", "danger", "유리 깨지는 소리 감지 - 위험"),
    "slam":       ("sound", "door_slam", "warning", "문 쾅 소리 감지 - 주의"),
    "keypad_ok":  ("keypad", "keypad_input", "normal", "정상 인증 완료 ({name}님 타건 리듬 일치 (유사도 91.2%))"),
    "keypad_bad": ("keypad", "keypad_anomaly", "warning", "비밀번호는 맞았지만 타건 패턴 불일치 (도용 의심 | 유사도 23.5%)"),
    "keypad_fail": ("keypad", "keypad_failed", "danger", "도어락 비밀번호 오류 경고음이 반복돼요"),
    "person":     ("vision", "person", None, "사람이 감지됐습니다 (1명)."),
    "entry":      ("vision", "entry", None, "현관으로 사람이 들어왔습니다 (1명)."),
    "threat":     ("vision", "threat", None, "사람이 양손을 머리 위로 들고 있습니다 (감지 2명)."),
    "fall":       ("vision", "fall", None, "사람이 쓰러진 형태로 보입니다 (감지 1명)."),
    "none":       ("vision", "none", None, "영상에서 사람이 보이지 않습니다."),
}

# 시나리오: (지연 초, 이벤트 종류)
SCENARIOS = {
    "homecoming": [(0, "keypad_ok"), (1.5, "entry")],          # 귀가 → 안전
    "intrusion":  [(0, "entry")],                               # 인증 없는 현관 진입 → 8초 후 주의
    "stolen_pin": [(0, "keypad_bad"), (1.5, "entry")],          # 도용 의심 + 진입 → 위험
    "scream":     [(0, "scream"), (1.0, "person")],              # 비명 → 사람 확인 → 위험
    "noise":      [(0, "glass"), (1.0, "none")],                 # 유리 소리, 영상 이상 없음 → 주의 유지
    "threat":     [(0, "slam"), (1.0, "threat")],                # 문 쾅 + 손 든 위협 자세 → 위험
    "fall":       [(0, "fall")],                                 # 쓰러짐 → 위험
}


def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _make_event(kind: str, name: str) -> dict:
    source, typ, risk, msg = EVENTS[kind]
    ev = {
        "device_id": DEVICE_ID,
        "event_id": f"demo-{uuid.uuid4().hex[:10]}",
        "source": source,
        "type": typ,
        "message": msg.format(name=name or "등록 사용자"),
        "confidence": 0.9,
        "timestamp": _now_iso(),
        "detail": {"demo": True},
    }
    if risk:
        ev["risk_level"] = risk
    if source == "vision":
        ev.update(scene=os.getenv("VISION_SCENE", "거실"), persons=[], behavior="")
    return ev


def _send(kind: str, name: str) -> None:
    from risk import judge
    judge.on_event(_make_event(kind, name))


def _user_name() -> str:
    try:
        u = database.get_user(session["user_id"])
        return (u or {}).get("name") or ""
    except Exception:
        return ""


def _camera_info() -> dict:
    info = {}
    for key in ("live", "capture"):
        p = CAMERA_DIR / f"{key}.jpg"
        if p.exists():
            m = p.stat().st_mtime
            info[key] = {"url": f"/static/camera/{key}.jpg?v={int(m * 1000)}", "age": round(time.time() - m, 1)}
    return info


@bp.get("/status")
@login_required
def status():
    from risk import judge
    data = judge.snapshot()
    data["camera"] = _camera_info()
    data["demo_enabled"] = DEMO_ENABLED
    data["server_time"] = _now_iso()
    return jsonify(data)


def _demo_guard():
    if not DEMO_ENABLED:
        return jsonify({"ok": False, "error": "시연 모드가 꺼져 있어요 (DEMO_MODE=0)"}), 403
    return None


@bp.post("/demo/event")
@login_required
def demo_event():
    if (g := _demo_guard()):
        return g
    kind = str((request.get_json(silent=True) or {}).get("kind", ""))
    if kind not in EVENTS:
        return jsonify({"ok": False, "error": f"알 수 없는 이벤트: {kind}"}), 400
    threading.Thread(target=_send, args=(kind, _user_name()), daemon=True).start()
    return jsonify({"ok": True, "kind": kind})


@bp.post("/demo/scenario")
@login_required
def demo_scenario():
    if (g := _demo_guard()):
        return g
    name = str((request.get_json(silent=True) or {}).get("name", ""))
    steps = SCENARIOS.get(name)
    if not steps:
        return jsonify({"ok": False, "error": f"알 수 없는 시나리오: {name}"}), 400
    user = _user_name()
    from risk import judge
    judge.reset(clear_log=False)   # 앞 시나리오의 열린 사건과 섞이지 않게 새 사건으로 시작

    def run():
        for delay, kind in steps:
            time.sleep(delay)
            _send(kind, user)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True, "name": name, "steps": [k for _, k in steps]})


@bp.post("/demo/clear")
@login_required
def demo_clear():
    if (g := _demo_guard()):
        return g
    from risk import judge
    judge.reset()
    try:
        n = database.clear_alerts()
    except Exception as e:
        return jsonify({"ok": False, "error": f"알림 삭제 실패: {e}"}), 500
    return jsonify({"ok": True, "deleted": n})
