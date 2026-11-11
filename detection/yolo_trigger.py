"""
detection/yolo_trigger.py — YOLO scene-state → conservative emit gate

Replaces pixel MotionTrigger. Fires only on meaningful person-state changes
(or external force / fall candidate), never on raw motion.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

try:
    from detection.yolo_detector import FrameDetections
except ImportError:
    from yolo_detector import FrameDetections


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class TriggerConfig:
    """Conservative defaults: fewer false alarms than demo-sensitive motion."""

    min_interval_sec: float = _env_float("YOLO_COOLDOWN_SEC", 8.0)
    warmup_frames: int = _env_int("YOLO_WARMUP_FRAMES", 10)
    persist_frames: int = _env_int("YOLO_PERSIST_FRAMES", 2)
    # fall: person bbox wider than tall + center in lower half (heuristic only)
    fall_aspect_wh: float = _env_float("YOLO_FALL_ASPECT", 1.15)
    fall_min_conf: float = _env_float("YOLO_FALL_MIN_CONF", 0.55)
    fall_cy_ratio_min: float = _env_float("YOLO_FALL_CY_MIN", 0.45)
    emit_on_appear: bool = True
    emit_on_count_up: bool = True
    emit_on_fall: bool = True
    # ROI as normalized x1,y1,x2,y2 (0..1). Empty = disabled.
    # Example: YOLO_ROI=0.0,0.0,0.35,1.0 for left doorway strip
    roi: tuple[float, float, float, float] | None = None
    emit_roi_enter: bool = True


@dataclass
class TriggerResult:
    fired: bool
    score: float
    reason: str
    person_count: int = 0
    fall_candidate: bool = False
    roi_hit: bool = False
    knife_hint: bool = False
    detail: dict[str, Any] = field(default_factory=dict)


def parse_roi(raw: str | None = None) -> tuple[float, float, float, float] | None:
    raw = (raw if raw is not None else os.getenv("YOLO_ROI", "")).strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(p) for p in parts)
    except ValueError:
        return None
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return None
    return x1, y1, x2, y2


class YoloTrigger:
    """Stateful gate over FrameDetections."""

    def __init__(self, config: TriggerConfig | None = None):
        self.cfg = config or TriggerConfig()
        if self.cfg.roi is None:
            self.cfg.roi = parse_roi()
        self._frame_i = 0
        self._streak_person = 0
        self._last_count = 0
        self._stable_count = 0
        self._last_fire_t = -1e9
        self._seen_ids: set[int] = set()
        self._in_roi_ids: set[int] = set()
        self._fall_streak = 0

    def reset(self) -> None:
        self._frame_i = 0
        self._streak_person = 0
        self._last_count = 0
        self._stable_count = 0
        self._last_fire_t = -1e9
        self._seen_ids.clear()
        self._in_roi_ids.clear()
        self._fall_streak = 0

    def _in_roi(self, det, fw: int, fh: int) -> bool:
        roi = self.cfg.roi
        if not roi or fw <= 0 or fh <= 0:
            return False
        x1, y1, x2, y2 = roi
        cx, cy = det.cx / fw, det.cy / fh
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _fall_person(self, dets: FrameDetections) -> bool:
        if not dets.persons or dets.frame_h <= 0:
            return False
        for p in dets.persons:
            if p.conf < self.cfg.fall_min_conf:
                continue
            if p.aspect_wh < self.cfg.fall_aspect_wh:
                continue
            if (p.cy / dets.frame_h) < self.cfg.fall_cy_ratio_min:
                continue
            return True
        return False

    def force_fire(self, now: float, score: float = 1.0, reason: str = "manual") -> TriggerResult:
        self._last_fire_t = now
        self._streak_person = 0
        self._fall_streak = 0
        return TriggerResult(
            True,
            score,
            reason,
            person_count=self._stable_count,
            detail={"forced": True},
        )

    def force_observe(
        self,
        dets: FrameDetections,
        now: float,
        reason: str = "external",
    ) -> TriggerResult:
        """Judge/manual 강제 관측: 현재 프레임 플래그를 채워 바로 mapper에 넘긴다."""
        tr = self.force_fire(now, score=dets.max_person_conf, reason=reason)
        tr.person_count = dets.person_count
        tr.fall_candidate = self._fall_person(dets)
        tr.knife_hint = any(o.cls_id == 43 for o in dets.others)
        tr.roi_hit = bool(
            self.cfg.roi
            and any(self._in_roi(p, dets.frame_w, dets.frame_h) for p in dets.persons)
        )
        tr.detail["detections"] = dets.as_dict()
        return tr

    def update(self, dets: FrameDetections, now: float) -> TriggerResult:
        self._frame_i += 1
        count = dets.person_count
        score = dets.max_person_conf
        knife = any(o.label == "knife" or o.cls_id == 43 for o in dets.others)
        fall = self._fall_person(dets)
        if fall:
            self._fall_streak += 1
        else:
            self._fall_streak = 0
        fall_ok = fall and self._fall_streak >= self.cfg.persist_frames

        if count > 0:
            self._streak_person += 1
        else:
            self._streak_person = 0
            # lose tracks when empty
            if self._last_count > 0 and self._streak_person == 0:
                pass

        # wait for persist before treating count as stable
        stable = count if self._streak_person >= self.cfg.persist_frames else self._stable_count
        if self._streak_person >= self.cfg.persist_frames:
            self._stable_count = count

        roi_enter = False
        new_ids: list[int] = []
        if dets.persons:
            for p in dets.persons:
                if p.track_id is not None:
                    if p.track_id not in self._seen_ids and self._streak_person >= self.cfg.persist_frames:
                        new_ids.append(p.track_id)
                    if self._in_roi(p, dets.frame_w, dets.frame_h):
                        if p.track_id not in self._in_roi_ids and self.cfg.roi:
                            roi_enter = True
                        self._in_roi_ids.add(p.track_id)
                    if self._streak_person >= self.cfg.persist_frames:
                        self._seen_ids.add(p.track_id)
            # also ROI without track ids: count-based enter
            if self.cfg.roi and not any(p.track_id is not None for p in dets.persons):
                in_now = any(self._in_roi(p, dets.frame_w, dets.frame_h) for p in dets.persons)
                if in_now and self._streak_person == self.cfg.persist_frames:
                    roi_enter = True

        if self._frame_i <= self.cfg.warmup_frames:
            self._last_count = self._stable_count
            return TriggerResult(False, score, "warmup", person_count=count, knife_hint=knife)

        # decide reason (priority: fall > roi > appear/count)
        reason = ""
        if self.cfg.emit_on_fall and fall_ok:
            reason = "fall_candidate"
        elif self.cfg.emit_roi_enter and roi_enter:
            reason = "roi_enter"
        elif self.cfg.emit_on_appear and self._last_count == 0 and stable >= 1:
            reason = "person_appear"
        elif self.cfg.emit_on_count_up and stable > self._last_count:
            reason = "person_count_up"
        elif knife and stable >= 1 and self._last_count == 0:
            # knife alone never fires; only with a fresh person appear already covered
            reason = ""

        if not reason:
            self._last_count = stable
            return TriggerResult(
                False,
                score,
                "stable" if stable else "no_person",
                person_count=stable,
                fall_candidate=fall_ok,
                roi_hit=bool(self.cfg.roi and any(
                    self._in_roi(p, dets.frame_w, dets.frame_h) for p in dets.persons
                )),
                knife_hint=knife,
            )

        if now - self._last_fire_t < self.cfg.min_interval_sec:
            self._last_count = stable
            return TriggerResult(
                False,
                score,
                "cooldown",
                person_count=stable,
                fall_candidate=fall_ok,
                knife_hint=knife,
                detail={"blocked_reason": reason},
            )

        self._last_fire_t = now
        self._last_count = stable
        return TriggerResult(
            True,
            score,
            reason,
            person_count=stable,
            fall_candidate=fall_ok,
            roi_hit=(reason == "roi_enter"),
            knife_hint=knife,
            detail={"new_track_ids": new_ids},
        )
