"""홈캠 영상 실시간 관측 서버 (경량 모션 트리거 → 클립 관측)

설계
----
- 상시: OpenCV로 프레임 수신 + MotionTrigger(저해상도 차분)만 수행
- 트리거 시: 링버퍼 클립 저장 후 video_analyzer.analyze_clip() 호출
  · mode=stub  : fixtures JSON (시연 기본, GPU/지연 없음)
  · mode=local : Qwen2.5-VL 실추론 (scripts/run_local.py 와 동일 API)
- 위험도 판정은 하지 않음. 관측 이벤트만 judge / 로그 / HTTP 로 전달

사용법 (프로젝트 루트 또는 detection/ 에서)
  python -m detection.vision --source clips/demo1.mp4 --mode stub
  python -m detection.vision --source 0 --mode stub --preview
  python -m detection.vision --source clips/demo2.mp4 --mode stub ^
      --stub-path fixtures/sample_result2.json

시연 팁
  - 기본 mode=stub. 발표 중 local 전환 금지 권장
  - --fixture-rotate 로 트리거마다 sample_result1..4 순환
  - --preview 창에서 t = 수동 트리거, q = 종료
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

try:
    import requests
except ImportError:  # HTTP 전송은 옵션
    requests = None  # type: ignore[assignment]

# ── 경로: 프로젝트 루트를 import path 에 추가 ──
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.motion_trigger import MotionTrigger, TriggerConfig  # noqa: E402
from video_analyzer import analyze_clip  # noqa: E402

# ───────────────────────── 기본 설정 ─────────────────────────
CLIP_SEC = 4.0                 # 트리거 시 저장할 링버퍼 길이(초)
PROCESS_EVERY_N = 2            # 모션 판정은 N프레임마다 (CPU 절약)
COOLDOWN_SEC = 8.0             # 트리거 쿨다운 (MotionTrigger 와 맞춤)
SERVER_URL = os.getenv("VISION_SERVER_URL", "")  # 예: http://localhost:5000/api/events
EVENT_LOG = Path(__file__).resolve().parent / "vision_events.jsonl"
CLIP_DIR = Path(__file__).resolve().parent / "clips_out"
DEFAULT_FIXTURES = [
    ROOT / "fixtures" / "sample_result1.json",
    ROOT / "fixtures" / "sample_result2.json",
    ROOT / "fixtures" / "sample_result3.json",
    ROOT / "fixtures" / "sample_result4.json",
]

stop = threading.Event()


# ───────────────────────── 유틸 ─────────────────────────
def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def ensure_message_str(message: Any) -> str:
    if isinstance(message, list):
        return " ".join(str(x) for x in message if x)
    if message is None:
        return ""
    return str(message)


def open_capture(source: str):
    """source: 웹캠 인덱스("0"), 파일 경로, rtsp://..."""
    import cv2

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {source}")
    return cap


def probe_fps(cap, fallback: float = 15.0) -> float:
    fps = float(cap.get(5))  # cv2.CAP_PROP_FPS
    if fps <= 1e-3 or fps > 120:
        return fallback
    return fps


class FrameRing:
    """최근 clip_sec 초 분량 BGR 프레임 보관."""

    def __init__(self, clip_sec: float, fps: float):
        self.fps = max(fps, 1.0)
        self.maxlen = max(int(clip_sec * self.fps), 1)
        self.frames: deque[np.ndarray] = deque(maxlen=self.maxlen)

    def push(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def __len__(self) -> int:
        return len(self.frames)

    def save_mp4(self, path: Path, fps: float | None = None) -> Path:
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.frames:
            raise RuntimeError("링버퍼가 비어 있어 클립을 저장할 수 없습니다.")
        h, w = self.frames[0].shape[:2]
        out_fps = float(fps or self.fps)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            out_fps,
            (w, h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter 실패: {path}")
        for fr in self.frames:
            writer.write(fr)
        writer.release()
        return path


class FixtureRotator:
    def __init__(self, paths: list[Path]):
        existing = [p for p in paths if p.exists()]
        if not existing:
            raise FileNotFoundError(
                "사용 가능한 fixture 가 없습니다. fixtures/sample_result*.json 을 확인하세요."
            )
        self.paths = existing
        self.i = 0

    def next_path(self) -> Path:
        p = self.paths[self.i % len(self.paths)]
        self.i += 1
        return p


def resolve_stub_path(
    *,
    stub_path: str | None,
    rotate: FixtureRotator | None,
    source: str,
) -> Path | None:
    """stub 모드에서 쓸 fixture 경로 결정."""
    if stub_path:
        return Path(stub_path)
    if rotate is not None:
        return rotate.next_path()
    # 파일명 힌트: demo2.mp4 → sample_result2.json
    name = Path(source).stem.lower()
    for key, fx in (
        ("1", "sample_result1.json"),
        ("2", "sample_result2.json"),
        ("3", "sample_result3.json"),
        ("4", "sample_result4.json"),
        ("demo1", "sample_result1.json"),
        ("demo2", "sample_result2.json"),
        ("demo3", "sample_result3.json"),
        ("demo4", "sample_result4.json"),
    ):
        if key in name:
            cand = ROOT / "fixtures" / fx
            if cand.exists():
                return cand
    default = ROOT / "fixtures" / "sample_result1.json"
    return default if default.exists() else None


def make_event(
    *,
    device_id: str,
    event_id: str,
    obs: dict[str, Any],
    motion_score: float,
    trigger_reason: str,
    clip_path: str,
    mode: str,
    stub_used: str | None,
) -> dict[str, Any]:
    """Judge / 앱 파이프라인으로 넘길 관측 이벤트 (위험 단정 없음)."""
    message = ensure_message_str(obs.get("message"))
    return {
        "device_id": device_id,
        "event_id": event_id,
        "source": "video",
        "type": "clip_observation",
        "message": message,
        "confidence": float(obs.get("confidence") or 0.0),
        "timestamp": obs.get("timestamp") or now_iso(),
        "scene": obs.get("scene") or "",
        "persons": obs.get("persons") or [],
        "clip_id": obs.get("clip_id") or event_id,
        "detail": {
            "motion_score": round(float(motion_score), 4),
            "trigger": trigger_reason,
            "clip_path": clip_path,
            "analyze_mode": mode,
            "stub_path": stub_used,
            "obs": {
                "source": "video",
                "scene": obs.get("scene") or "",
                "persons": obs.get("persons") or [],
                "message": message,
                "confidence": float(obs.get("confidence") or 0.0),
                "clip_id": obs.get("clip_id") or event_id,
                "timestamp": obs.get("timestamp") or now_iso(),
            },
        },
    }


def notify(event: dict[str, Any]) -> None:
    """jsonl 기록 → risk.judge.on_event → (옵션) HTTP."""
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    try:
        from risk.judge import on_event

        on_event(event)
    except Exception as e:  # judge 미구현·DB 없음 등 → 시연이 죽지 않게
        print(f"[vision] judge.on_event 스킵/실패: {e}")

    if SERVER_URL:
        if requests is None:
            print("[vision] requests 미설치 → HTTP 전송 스킵")
        else:
            try:
                requests.post(SERVER_URL, json=event, timeout=3)
            except Exception as e:
                print(f"[vision] 서버 전송 실패: {e}")


def run_observation(
    *,
    clip_path: Path,
    mode: str,
    stub_path: Path | None,
    device_id: str,
    motion_score: float,
    trigger_reason: str,
    analyze_sleep: float,
) -> dict[str, Any]:
    """
    클립 관측 실행.
    stub: fixtures 반환 (시연). local: Qwen 실추론 (run_local 과 동일 함수).
    """
    event_id = f"vid-{uuid.uuid4().hex[:10]}"
    stub_arg = str(stub_path) if stub_path else None

    # 시연 연출: stub 에서도 짧은 대기 → "분석 중"처럼 보이게
    if analyze_sleep > 0:
        time.sleep(analyze_sleep)

    t0 = time.time()
    obs = analyze_clip(
        str(clip_path),
        mode=mode,  # type: ignore[arg-type]
        stub_path=stub_arg,
    )
    # 실시간 사건처럼 보이게 타임스탬프·clip_id 갱신
    obs["timestamp"] = now_iso()
    obs["clip_id"] = obs.get("clip_id") or event_id
    obs["message"] = ensure_message_str(obs.get("message"))
    infer_sec = time.time() - t0

    event = make_event(
        device_id=device_id,
        event_id=event_id,
        obs=dict(obs),
        motion_score=motion_score,
        trigger_reason=trigger_reason,
        clip_path=str(clip_path),
        mode=mode,
        stub_used=stub_arg,
    )
    print(
        f"  ✓ 관측 완료 ({mode}, {infer_sec:.2f}s) "
        f"scene={event['scene']!r} persons={len(event['persons'])} "
        f"msg={event['message'][:60]}..."
    )
    notify(event)
    return event


# ───────────────────────── 메인 루프 ─────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="실시간 영상 트리거 → 클립 관측 서버")
    parser.add_argument(
        "--source",
        default="0",
        help="웹캠 인덱스(0) | mp4 경로 | rtsp://...",
    )
    parser.add_argument("--device-id", default="homecam-1")
    parser.add_argument(
        "--mode",
        choices=("stub", "local"),
        default=os.getenv("VIDEO_ANALYZER_MODE", "stub"),
        help="analyze_clip 모드. 시연 기본 stub",
    )
    parser.add_argument("--stub-path", default=None, help="고정 fixture JSON 경로")
    parser.add_argument(
        "--fixture-rotate",
        action="store_true",
        help="트리거마다 fixtures/sample_result1..4 순환",
    )
    parser.add_argument("--clip-sec", type=float, default=CLIP_SEC)
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_SEC)
    parser.add_argument("--score-threshold", type=float, default=0.02)
    parser.add_argument("--pixel-threshold", type=int, default=25)
    parser.add_argument(
        "--analyze-sleep",
        type=float,
        default=0.8,
        help="stub 시연용 가짜 추론 대기(초). local 은 무시해도 됨",
    )
    parser.add_argument("--preview", action="store_true", help="디버그 미리보기 창")
    parser.add_argument(
        "--loop-file",
        action="store_true",
        help="파일 소스 끝날 때 처음으로 되감기 (반복 시연)",
    )
    parser.add_argument(
        "--no-realtime-file",
        action="store_true",
        help="파일 소스를 FPS 대기 없이 최대한 빠르게 재생",
    )
    args = parser.parse_args()
    pace_file = not args.no_realtime_file

    import cv2

    cap = open_capture(args.source)
    fps = probe_fps(cap)
    ring = FrameRing(args.clip_sec, fps)
    trigger = MotionTrigger(
        TriggerConfig(
            score_threshold=args.score_threshold,
            pixel_threshold=args.pixel_threshold,
            min_interval_sec=args.cooldown,
        )
    )
    rotator = FixtureRotator(DEFAULT_FIXTURES) if args.fixture_rotate else None
    is_file = (not args.source.isdigit()) and Path(args.source).exists()

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("영상 실시간 관측 서버 시작")
    print(f"  source={args.source}  fps≈{fps:.1f}  mode={args.mode}")
    print(f"  trigger={trigger.as_dict()}")
    print(f"  event_log={EVENT_LOG}")
    if args.preview:
        print("  preview: t=수동트리거, q=종료")
    print("=" * 70)

    frame_i = 0
    try:
        while not stop.is_set():
            loop_t0 = time.monotonic()
            ok, frame = cap.read()
            if not ok or frame is None:
                if is_file and args.loop_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    trigger.reset()
                    print("[vision] 파일 끝 → 루프")
                    continue
                print("[vision] 프레임 없음 → 종료")
                break

            ring.push(frame)
            frame_i += 1
            now = time.time()
            fired = False
            tr_score = 0.0
            tr_reason = ""

            # 모션 판정 (일부 프레임만)
            if frame_i % PROCESS_EVERY_N == 0:
                result = trigger.update(frame, now)
                tr_score, tr_reason = result.score, result.reason
                if frame_i % max(int(fps), 1) == 0:
                    print(
                        f"[{now_iso()}] motion={tr_score:.3f} state={tr_reason}"
                    )
                fired = result.fired

            # 미리보기 / 수동 트리거
            if args.preview:
                vis = frame.copy()
                cv2.putText(
                    vis,
                    f"motion={tr_score:.3f} {tr_reason}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("vision-trigger", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("t"):
                    forced = trigger.force_fire(now, score=tr_score)
                    fired = True
                    tr_score, tr_reason = forced.score, forced.reason

            if fired:
                print(
                    f"[{now_iso()}] ⚡ TRIGGER ({tr_reason}) "
                    f"score={tr_score:.3f} → 클립 관측"
                )
                clip_path = CLIP_DIR / f"clip_{int(now)}_{uuid.uuid4().hex[:6]}.mp4"
                try:
                    ring.save_mp4(clip_path, fps=fps)
                except Exception as e:
                    # 버퍼 실패 시 빈 임시 파일이라도 stub 은 동작
                    print(f"[vision] 클립 저장 실패({e}) → 임시 경로로 stub 진행")
                    clip_path = Path(tempfile.gettempdir()) / f"vision_stub_{uuid.uuid4().hex}.mp4"
                    clip_path.write_bytes(b"")

                stub = None
                if args.mode == "stub":
                    stub = resolve_stub_path(
                        stub_path=args.stub_path,
                        rotate=rotator,
                        source=args.source,
                    )
                    print(f"  stub_path={stub}")

                try:
                    run_observation(
                        clip_path=clip_path,
                        mode=args.mode,
                        stub_path=stub,
                        device_id=args.device_id,
                        motion_score=tr_score,
                        trigger_reason=tr_reason,
                        analyze_sleep=args.analyze_sleep if args.mode == "stub" else 0.0,
                    )
                except Exception as e:
                    print(f"[vision] 관측 실패: {e}")

            # 파일 소스는 실시간 속도로 재생
            if is_file and pace_file:
                delay = (1.0 / fps) - (time.monotonic() - loop_t0)
                if delay > 0:
                    time.sleep(delay)

    except KeyboardInterrupt:
        print("\n[vision] 사용자 중단")
    finally:
        stop.set()
        cap.release()
        if args.preview:
            try:
                import cv2 as _cv2

                _cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()
