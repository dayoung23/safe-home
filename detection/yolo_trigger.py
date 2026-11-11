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
    # fall: 1순위 pose 키포인트 몸통 각도 / 2순위 박스 비율 (화면 끝에 잘린 사람은 판정 안 함)
    fall_aspect_wh: float = _env_float("YOLO_FALL_ASPECT", 1.15)
    fall_min_conf: float = _env_float("YOLO_FALL_MIN_CONF", 0.45)  # 누운 사람은 탐지 신뢰도가 낮게 나옴
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

    def _lying(self, p, fw: int, fh: int) -> bool | None:
        """누운 자세면 True, 서/앉음이면 False, 판단 불가면 None.

        1순위: pose 키포인트 — 어깨 중점→엉덩이 중점(몸통)이 수평에 가까우면 누움
        2순위: 엉덩이가 안 보이면 박스 비율 — 단, 화면 끝에 닿은 사람(캠 앞 근접 상반신)은 판단 안 함
               (노트북 캠 앞에 앉으면 머리·어깨만 가로로 넓게 잡혀 쓰러짐으로 오인되던 문제 방지)
        """
        k = p.kpts
        if k is not None and len(k) >= 13:
            sh = [k[i, :2] for i in (5, 6) if k[i, 2] > 0.3]
            hip = [k[i, :2] for i in (11, 12) if k[i, 2] > 0.3]
            if sh and hip:
                s_c = sum(sh) / len(sh)
                h_c = sum(hip) / len(hip)
                dx, dy = abs(float(h_c[0] - s_c[0])), abs(float(h_c[1] - s_c[1]))
                if (dx * dx + dy * dy) ** 0.5 < 0.05 * min(fw, fh):
                    return None          # 몸통이 너무 작게 잡힘 → 판단 불가
                return dx > dy           # 몸통이 45도보다 눕혀짐
        x1, y1, x2, y2 = p.xyxy
        mx, my = 0.02 * fw, 0.02 * fh
        if x1 < mx or y1 < my or x2 > fw - mx or y2 > fh - my:
            return None                  # 잘린 사람 → 판단 불가
        if p.aspect_wh < self.cfg.fall_aspect_wh:
            return False
        return (p.cy / fh) >= self.cfg.fall_cy_ratio_min

    def _fall_person(self, dets: FrameDetections) -> bool:
        if not dets.persons or dets.frame_h <= 0:
            return False
        for p in dets.persons:
            if p.conf < self.cfg.fall_min_conf:
                continue
            if self._lying(p, dets.frame_w, dets.frame_h) is True:
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
