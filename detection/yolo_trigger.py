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
    # 손 들기(위협·협박 자세): 양 손목이 머리(코) 위에 있는 프레임이 연속 N번
    hands_up_frames: int = _env_int("YOLO_HANDSUP_FRAMES", 3)
    emit_on_hands_up: bool = True
    emit_on_appear: bool = True
    emit_on_count_up: bool = True
    emit_on_fall: bool = True
    # 현관 구역(ROI) x1,y1,x2,y2 (0~1 비율). 비우면 사용 안 함.
    # 예: YOLO_ROI=0.0,0.0,0.3,1.0  → 화면 왼쪽 30% 를 현관으로 간주
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
    hands_up: bool = False
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
        self._hands_streak = 0
        self._roi_streak = 0
        self._last_fire_by: dict[str, float] = {}

    def reset(self) -> None:
        self._frame_i = 0
        self._streak_person = 0
        self._last_count = 0
        self._stable_count = 0
        self._last_fire_t = -1e9
        self._seen_ids.clear()
        self._in_roi_ids.clear()
        self._fall_streak = 0
        self._hands_streak = 0
        self._roi_streak = 0
        self._last_fire_by = {}

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

    @staticmethod
    def _hands_up_person(p) -> bool:
        """양 손목(9,10)이 머리 높이(코, 없으면 어깨보다 위)보다 위에 있으면 손 든 자세."""
        k = p.kpts
        if k is None or len(k) < 11:
            return False
        if k[9, 2] < 0.3 or k[10, 2] < 0.3:
            return False
        if k[0, 2] > 0.3:
            head_y = float(k[0, 1])                         # 코
        elif k[5, 2] > 0.3 and k[6, 2] > 0.3:
            head_y = float(min(k[5, 1], k[6, 1])) - 0.15 * p.height   # 어깨보다 조금 위
        else:
            return False
        return float(k[9, 1]) < head_y and float(k[10, 1]) < head_y

    def _hands_up(self, dets: FrameDetections) -> bool:
        return any(self._hands_up_person(p) for p in dets.persons)

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
        tr.hands_up = self._hands_up(dets)
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
        hands = self._hands_up(dets)
        self._hands_streak = self._hands_streak + 1 if hands else 0
        hands_ok = hands and self._hands_streak >= max(self.cfg.hands_up_frames, 1)
        # 현관 구역: "구역 안에 사람 있음" 상태가 persist 프레임 이어진 순간 = 진입 (추적 기능 없이도 동작)
        in_roi_now = bool(self.cfg.roi) and any(
            self._in_roi(p, dets.frame_w, dets.frame_h) for p in dets.persons)
        self._roi_streak = self._roi_streak + 1 if in_roi_now else 0

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

        roi_enter = self._roi_streak == self.cfg.persist_frames
        new_ids: list[int] = []
        if dets.persons:
            for p in dets.persons:
                if p.track_id is not None:
                    if p.track_id not in self._seen_ids and self._streak_person >= self.cfg.persist_frames:
                        new_ids.append(p.track_id)
                    if self._streak_person >= self.cfg.persist_frames:
                        self._seen_ids.add(p.track_id)

        if self._frame_i <= self.cfg.warmup_frames:
            self._last_count = self._stable_count
            return TriggerResult(False, score, "warmup", person_count=count, knife_hint=knife)

        # decide reason (priority: fall > hands_up > roi > appear/count)
        reason = ""
        if self.cfg.emit_on_fall and fall_ok:
            reason = "fall_candidate"
        elif self.cfg.emit_on_hands_up and hands_ok:
            reason = "hands_up"
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
                roi_hit=in_roi_now,
                knife_hint=knife,
                hands_up=hands_ok,
            )

        # 쿨다운은 이유별로 따로 (사람 등장 직후에 손 들기·쓰러짐이 와도 막히지 않게)
        if now - self._last_fire_by.get(reason, -1e9) < self.cfg.min_interval_sec:
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
        self._last_fire_by[reason] = now
        self._last_count = stable
        return TriggerResult(
            True,
            score,
            reason,
            person_count=stable,
            fall_candidate=fall_ok,
            roi_hit=(reason == "roi_enter"),
            knife_hint=knife,
            hands_up=hands_ok,
            detail={"new_track_ids": new_ids},
        )
