"""경량 실시간 변화(모션) 트리거.

대형 VL을 상시 돌리지 않고, 다운스케일 그레이프레임 차분만으로
"클립 관측을 시작할 시점"을 판정한다.

점수: 리사이즈 → 그레이 → (선택) 블러 → |cur-prev| → 임계값 넘는 픽셀 비율
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TriggerConfig:
    """해커톤 시연용 기본값은 조금 민감하게 잡음 (데모에서 잘 터지게)."""

    width: int = 160
    height: int = 90
    pixel_threshold: int = 25          # 픽셀 밝기 차이 (0~255)
    score_threshold: float = 0.02      # 변화 픽셀 비율 (0~1)
    blur_ksize: int = 5                # 홀수. 0이면 블러 생략
    min_interval_sec: float = 8.0      # 트리거 쿨다운
    warmup_frames: int = 8             # 초기 프레임은 기준만 잡고 무시
    persist_frames: int = 2            # 연속 N프레임 넘어야 확정 (깜빡임 완화)


@dataclass
class TriggerResult:
    fired: bool
    score: float
    reason: str


class MotionTrigger:
    """상태 있는 모션 트리거. 프레임마다 update() 호출."""

    def __init__(self, config: TriggerConfig | None = None):
        self.cfg = config or TriggerConfig()
        self._prev: np.ndarray | None = None
        self._frame_i = 0
        self._streak = 0
        self._last_fire_t = -1e9

    def reset(self) -> None:
        self._prev = None
        self._frame_i = 0
        self._streak = 0
        self._last_fire_t = -1e9

    def prepare(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR/그레이 프레임 → 트리거용 소형 그레이 uint8."""
        import cv2

        cfg = self.cfg
        if frame_bgr.ndim == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr
        small = cv2.resize(gray, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
        if cfg.blur_ksize and cfg.blur_ksize > 1:
            k = cfg.blur_ksize if cfg.blur_ksize % 2 == 1 else cfg.blur_ksize + 1
            small = cv2.GaussianBlur(small, (k, k), 0)
        return small

    def score(self, frame_bgr: np.ndarray) -> float:
        """현재 프레임의 변화 점수만 계산 (상태 갱신 포함, 발화 판정은 안 함)."""
        cur = self.prepare(frame_bgr)
        self._frame_i += 1
        if self._prev is None:
            self._prev = cur
            return 0.0
        diff = np.abs(cur.astype(np.int16) - self._prev.astype(np.int16))
        changed = diff > self.cfg.pixel_threshold
        self._prev = cur
        return float(np.mean(changed))

    def update(self, frame_bgr: np.ndarray, now: float) -> TriggerResult:
        """
        프레임 1장을 넣고 트리거 여부를 반환.
        now: time.time() 등 초 단위 시각 (쿨다운용)
        """
        s = self.score(frame_bgr)

        if self._frame_i <= self.cfg.warmup_frames:
            self._streak = 0
            return TriggerResult(False, s, "warmup")

        if s >= self.cfg.score_threshold:
            self._streak += 1
        else:
            self._streak = 0
            return TriggerResult(False, s, "below_threshold")

        if self._streak < self.cfg.persist_frames:
            return TriggerResult(False, s, "persist")

        if now - self._last_fire_t < self.cfg.min_interval_sec:
            return TriggerResult(False, s, "cooldown")

        self._last_fire_t = now
        self._streak = 0
        return TriggerResult(True, s, "motion")

    def force_fire(self, now: float, score: float = 1.0) -> TriggerResult:
        """수동/외부 신호로 강제 발화 (쿨다운 갱신)."""
        self._last_fire_t = now
        self._streak = 0
        return TriggerResult(True, score, "manual")

    def as_dict(self) -> dict[str, Any]:
        c = self.cfg
        return {
            "width": c.width,
            "height": c.height,
            "pixel_threshold": c.pixel_threshold,
            "score_threshold": c.score_threshold,
            "min_interval_sec": c.min_interval_sec,
            "persist_frames": c.persist_frames,
        }
