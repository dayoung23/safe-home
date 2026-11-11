"""
detection/yolo_detector.py — Ultralytics YOLO wrapper (person-centric, conservative)

Stream frames in → structured detections out. No risk judgment.
Default model: yolo26n-pose.pt — 사람 박스 + 17개 관절(키포인트). 쓰러짐을 몸통 각도로 판정하기 위함.
(누운 사람 탐지가 yolov8n-pose 보다 확실함. 구버전 ultralytics 면 yolov8n-pose.pt 로 자동 전환)
Override with YOLO_MODEL env (일반 yolov8n.pt 도 동작하지만 쓰러짐은 박스 비율로만 판정).
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# COCO class ids we optionally care about (person is required path)
PERSON_CLS = 0
# knife is COCO 43 — only used as a weak "suspicious" hint, never alone as danger
KNIFE_CLS = 43

DEFAULT_MODEL = os.getenv("YOLO_MODEL", "yolo26n-pose.pt")
FALLBACK_MODEL = "yolov8n-pose.pt"
DEFAULT_CONF = float(os.getenv("YOLO_CONF", "0.50"))
DEFAULT_IOU = float(os.getenv("YOLO_IOU", "0.45"))
DEFAULT_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
DEFAULT_DEVICE = os.getenv("YOLO_DEVICE", "")  # "" = ultralytics auto


@dataclass
class Det:
    cls_id: int
    label: str
    conf: float
    xyxy: tuple[float, float, float, float]  # x1,y1,x2,y2
    track_id: int | None = None
    kpts: Any = None  # pose 모델일 때 (17, 3) = x, y, conf (COCO 키포인트), 아니면 None

    @property
    def cx(self) -> float:
        return 0.5 * (self.xyxy[0] + self.xyxy[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.xyxy[1] + self.xyxy[3])

    @property
    def width(self) -> float:
        return max(0.0, self.xyxy[2] - self.xyxy[0])

    @property
    def height(self) -> float:
        return max(0.0, self.xyxy[3] - self.xyxy[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_wh(self) -> float:
        """width / height. Standing person ~0.3–0.6; fallen often >= 1.0."""
        return self.width / self.height if self.height > 1e-3 else 0.0

    def zone(self, frame_w: int, frame_h: int) -> str:
        """Coarse screen zone for factual desc (no semantics)."""
        if frame_w <= 0 or frame_h <= 0:
            return "중앙"
        xr = self.cx / frame_w
        yr = self.cy / frame_h
        horiz = "좌측" if xr < 0.33 else ("우측" if xr > 0.66 else "중앙")
        vert = "상단" if yr < 0.33 else ("하단" if yr > 0.66 else "중단")
        if horiz == "중앙" and vert == "중단":
            return "중앙"
        return f"{horiz} {vert}"


@dataclass
class FrameDetections:
    persons: list[Det] = field(default_factory=list)
    others: list[Det] = field(default_factory=list)
    frame_w: int = 0
    frame_h: int = 0
    infer_ms: float = 0.0

    @property
    def person_count(self) -> int:
        return len(self.persons)

    @property
    def max_person_conf(self) -> float:
        return max((p.conf for p in self.persons), default=0.0)

    def as_dict(self) -> dict[str, Any]:
        def _one(d: Det) -> dict[str, Any]:
            return {
                "cls_id": d.cls_id,
                "label": d.label,
                "conf": round(d.conf, 4),
                "xyxy": [round(x, 1) for x in d.xyxy],
                "track_id": d.track_id,
            }

        return {
            "person_count": self.person_count,
            "persons": [_one(p) for p in self.persons],
            "others": [_one(o) for o in self.others],
            "infer_ms": round(self.infer_ms, 1),
        }


class YoloDetector:
    """Lazy-loads Ultralytics YOLO once; thread-safe infer."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        imgsz: int = DEFAULT_IMGSZ,
        device: str | None = None,
        use_track: bool | None = None,
    ):
        self.model_name = model_name
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device if device is not None else (DEFAULT_DEVICE or None)
        # tracking helps appear/leave; disable with YOLO_TRACK=0 on weak Pi
        if use_track is None:
            use_track = os.getenv("YOLO_TRACK", "1").strip() not in ("0", "false", "False")
        self.use_track = use_track
        self._model = None
        self._names: dict[int, str] = {}
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise ImportError(
                    "ultralytics 가 필요합니다: pip install ultralytics"
                ) from e
            print(f"[yolo] loading {self.model_name} (conf={self.conf}, imgsz={self.imgsz})")
            try:
                self._model = YOLO(self.model_name)
            except Exception as e:
                print(f"[yolo] {self.model_name} 로드 실패({e}) → {FALLBACK_MODEL}")
                self.model_name = FALLBACK_MODEL
                self._model = YOLO(self.model_name)
            names = getattr(self._model, "names", None) or {}
            if isinstance(names, dict):
                self._names = {int(k): str(v) for k, v in names.items()}
            else:
                self._names = {i: str(n) for i, n in enumerate(names)}
            print(f"[yolo] ready (track={self.use_track})")

    def _parse(self, result, frame_w: int, frame_h: int, infer_ms: float) -> FrameDetections:
        out = FrameDetections(frame_w=frame_w, frame_h=frame_h, infer_ms=infer_ms)
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        kxy = kcf = None
        kp = getattr(result, "keypoints", None)
        if kp is not None and getattr(kp, "xy", None) is not None:
            try:
                kxy = kp.xy.cpu().numpy()
                kcf = kp.conf.cpu().numpy() if kp.conf is not None else None
            except Exception:
                kxy = kcf = None
        ids = None
        if getattr(boxes, "id", None) is not None:
            try:
                ids = boxes.id.cpu().numpy().astype(int)
            except Exception:
                ids = None
        for i in range(len(clss)):
            cid = int(clss[i])
            c = float(confs[i])
            if c < self.conf:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            tid = int(ids[i]) if ids is not None else None
            kpts = None
            if kxy is not None and i < len(kxy):
                conf_col = kcf[i][:, None] if kcf is not None else np.ones((kxy.shape[1], 1))
                kpts = np.concatenate([kxy[i], conf_col], axis=1)
            det = Det(
                cls_id=cid,
                label=self._names.get(cid, str(cid)),
                conf=c,
                xyxy=(x1, y1, x2, y2),
                track_id=tid,
                kpts=kpts,
            )
            if cid == PERSON_CLS:
                out.persons.append(det)
            elif cid == KNIFE_CLS:
                out.others.append(det)
        # stable order: larger / more confident first
        out.persons.sort(key=lambda d: (-d.conf, -d.area))
        return out

    def detect(self, frame_bgr: np.ndarray) -> FrameDetections:
        """Run one-frame detection (or track)."""
        self.load()
        import time

        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = dict(
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        if self.device:
            kwargs["device"] = self.device
        with self._lock:
            if self.use_track:
                try:
                    results = self._model.track(frame_bgr, persist=True, **kwargs)
                except Exception:
                    results = self._model.predict(frame_bgr, **kwargs)
            else:
                results = self._model.predict(frame_bgr, **kwargs)
        ms = (time.perf_counter() - t0) * 1000.0
        result = results[0] if results else None
        if result is None:
            return FrameDetections(frame_w=w, frame_h=h, infer_ms=ms)
        return self._parse(result, w, h, ms)
