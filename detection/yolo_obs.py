"""
detection/yolo_obs.py — YOLO detections + trigger → Judge observation dict

Conservative mapping:
  fall_candidate → type=fall
  roi_enter (+ person) → type=suspicious  (never auto-intruder)
  knife + person → type=suspicious
  person otherwise → type=person
  empty (forced check) → type=none

gender always 불명. scene from VISION_SCENE env or "".
No risk wording — factual templates only.
"""
from __future__ import annotations

import os
from typing import Any

try:
    from detection.yolo_detector import FrameDetections
    from detection.yolo_trigger import TriggerResult
except ImportError:
    from yolo_detector import FrameDetections
    from yolo_trigger import TriggerResult

VISION_TYPES = ("person", "fall", "threat", "entry", "intruder", "suspicious", "fire", "motion", "none")
DEFAULT_SCENE = os.getenv("VISION_SCENE", "")


def _person_entries(dets: FrameDetections) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for p in dets.persons:
        zone = p.zone(dets.frame_w, dets.frame_h)
        tid = f", track={p.track_id}" if p.track_id is not None else ""
        wh = f"가로세로비 {p.aspect_wh:.2f}"
        out.append({
            "gender": "불명",
            "desc": f"{zone}에 있는 사람 (신뢰도 {p.conf:.2f}, {wh}{tid})",
        })
    return out


def map_observation(
    dets: FrameDetections,
    trigger: TriggerResult,
    scene: str | None = None,
) -> dict[str, Any]:
    """Build observation dict compatible with judge / legacy clip_analyzer.normalize."""
    scene = DEFAULT_SCENE if scene is None else scene
    persons = _person_entries(dets)
    n = len(persons)
    conf = float(dets.max_person_conf if n else (trigger.score or 0.0))
    conf = min(max(conf, 0.0), 1.0)

    # Conservative type selection (priority order)
    if trigger.fall_candidate or trigger.reason == "fall_candidate":
        vtype = "fall"
        behavior = "몸통이 수평에 가까운 누운 자세가 연속 감지됨 (낙상 후보)"
        message = f"사람이 쓰러진 형태로 보입니다 (감지 {n}명)." if n else "낙상 후보가 감지됐습니다."
    elif trigger.hands_up or trigger.reason == "hands_up":
        vtype = "threat"
        behavior = "양손을 머리 위로 든 자세가 연속 감지됨 (협박·위협 상황 후보)"
        message = f"사람이 양손을 머리 위로 들고 있습니다 (감지 {n}명)."
    elif trigger.reason == "roi_enter" or (
        trigger.roi_hit and trigger.reason in ("person_appear", "person_count_up")
    ):
        vtype = "entry"
        behavior = "현관(지정 구역)으로 사람이 들어옴"
        message = f"현관으로 사람이 들어왔습니다 ({n}명)."
    elif trigger.knife_hint and n >= 1:
        vtype = "suspicious"
        behavior = "사람과 함께 knife 객체가 함께 탐지됨 (약한 단서)"
        message = f"사람과 날카로운 물체 후보가 함께 보입니다 (감지 {n}명)."
    elif n >= 1:
        vtype = "person"
        if trigger.reason == "person_count_up":
            behavior = f"사람 수 증가 → {n}명"
            message = f"사람이 {n}명 감지됐습니다."
        elif trigger.reason == "person_appear":
            behavior = "사람 등장"
            message = f"사람이 감지됐습니다 ({n}명)."
        else:
            behavior = "사람 감지"
            message = f"사람이 감지됐습니다 ({n}명)."
    else:
        vtype = "none"
        behavior = "사람 박스 없음"
        message = "영상에서 사람이 보이지 않습니다."

    if vtype not in VISION_TYPES:
        vtype = "person" if n else "none"

    return {
        "type": vtype,
        "scene": scene or "",
        "persons": persons,
        "behavior": behavior,
        "message": message,
        "confidence": conf,
    }
