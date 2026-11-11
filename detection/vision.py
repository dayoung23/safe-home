"""
detection/vision.py — 홈캠 영상 실시간 서버 (트리거 → 클립 관측 → judge)

  평상시 : 프레임만 받고 MotionTrigger(160x90 흑백 차분)만 돌림 → 추론 없음, CPU 거의 안 씀
  트리거 : ① 움직임 감지  ② judge 의 확인 요청(위험 소리·키패드 이상 → request_check)
           → 트리거 전 CLIP_SEC + 이후 POST_SEC 를 클립으로 묶어 분석 스레드에 넘김
  분석   : clip_analyzer (yolo = YOLO26 pose 실추론, 기본 / stub = 미리 추론한 fixture, 비상용
           / local = Qwen2.5-VL 상세 설명)
           분석은 별도 스레드라 추론이 오래 걸려도 프레임 루프는 멈추지 않음.
           분석 중에 들어온 트리거는 건너뜀 (GPU 1장 기준).
  전달   : 관측 이벤트 dict → notify() → vision_events.jsonl + risk.judge.on_event (+옵션 HTTP)

위험 등급은 정하지 않는다 (judge 담당). 클립은 local 추론에만 임시 파일로 쓰고 바로 지운다.

실행
  python main.py                                   # 서버 + 오디오 + 영상 같이 (권장)
  python -m detection.vision --source 0 --preview  # 영상만 단독 (t = 수동 트리거, q = 종료)
  python -m detection.vision --source http://<파이IP>:8000/stream.mjpg --preview   # 라즈베리파이 카메라
  python -m detection.vision --source demo.mp4 --mode local
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import queue
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

try:
    from detection.motion_trigger import MotionTrigger, TriggerConfig
    from detection.clip_analyzer import analyze_clip, analyze_frames, list_fixtures, load_yolo
except ImportError:   # detection 폴더 안에서 단독 실행할 때
    from motion_trigger import MotionTrigger, TriggerConfig
    from clip_analyzer import analyze_clip, analyze_frames, list_fixtures, load_yolo

# ───────────────────────── 설정 ─────────────────────────
CLIP_SEC = 4.0           # 트리거 이전 구간 (링버퍼)
POST_SEC = 2.0           # 트리거 이후 더 모을 구간 (yolo / local)
PROCESS_EVERY_N = 2      # 모션 판정은 N프레임마다
COOLDOWN_SEC = 8.0       # 모션 트리거 쿨다운
STUB_DELAY_SEC = 0.8     # stub 모드 시연 연출용 대기 ("분석 중" 표시)
SERVER_URL = os.getenv("VISION_SERVER_URL", "")
FIXED_FIXTURE = os.getenv("VISION_FIXTURE", "")   # 비우면 fixtures/*.json 을 파일명 순서대로 순환
EVENT_LOG = Path(__file__).resolve().parent / "vision_events.jsonl"

stop = threading.Event()
_jobs: "queue.Queue[dict]" = queue.Queue(maxsize=1)
_busy = threading.Event()
_check_lock = threading.Lock()
_check_reason: str | None = None


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


# ───────────────────────── judge → vision 확인 요청 ─────────────────────────
def request_check(reason: str = "external") -> None:
    """judge 가 위험 소리·키패드 이상을 받았을 때 호출 → 다음 프레임에서 강제 트리거."""
    global _check_reason
    with _check_lock:
        _check_reason = reason


def _pop_check() -> str | None:
    global _check_reason
    with _check_lock:
        r, _check_reason = _check_reason, None
    return r


# ───────────────────────── 클립 / fixture ─────────────────────────
def save_clip(frames: list[np.ndarray], fps: float) -> Path:
    import cv2

    path = Path(tempfile.gettempdir()) / f"safehome_clip_{uuid.uuid4().hex[:8]}.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter 실패: {path}")
    for fr in frames:
        writer.write(fr)
    writer.release()
    return path


class FixtureCycle:
    def __init__(self):
        self._it = None

    def next(self) -> Path | None:
        if FIXED_FIXTURE:
            return Path(FIXED_FIXTURE)
        if self._it is None:
            files = list_fixtures()
            if not files:
                return None
            self._it = itertools.cycle(files)
        return next(self._it)


_fixtures = FixtureCycle()


# ───────────────────────── 이벤트 · 전달 ─────────────────────────
def make_event(device_id: str, obs: dict[str, Any], job: dict, mode: str,
               fixture: str | None, infer_sec: float) -> dict[str, Any]:
    """공통 필드(device_id, event_id, source, type, message, confidence, timestamp) + 영상 전용 필드."""
    return {
        "device_id": device_id,
        "event_id": f"vision-{uuid.uuid4().hex[:10]}",
        "source": "vision",
        "type": obs["type"],
        "message": obs["message"],
        "confidence": obs["confidence"],
        "timestamp": now_iso(),
        "scene": obs["scene"],
        "persons": obs["persons"],
        "behavior": obs["behavior"],
        "detail": {
            "trigger": job["reason"],          # motion | sound | keypad | manual
            "trigger_time": job["timestamp"],
            "motion_score": round(job["score"], 4),
            "analyze_mode": mode,
            "fixture": fixture,
            "infer_sec": round(infer_sec, 2),
        },
    }


def notify(event: dict[str, Any]) -> None:
    with open(EVENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    try:
        from risk import judge
        judge.on_event(event)          # 이미 분석 스레드 안이라 프레임 루프는 안 막힘
    except Exception as e:
        print(f"[vision] judge.on_event 실패: {e}")
    if SERVER_URL:
        try:
            import requests
            requests.post(SERVER_URL, json=event, timeout=3)
        except Exception as e:
            print(f"[vision] 서버 전송 실패: {e}")


# ───────────────────────── 분석 스레드 ─────────────────────────
def _analysis_worker(device_id: str, mode: str) -> None:
    while not stop.is_set():
        try:
            job = _jobs.get(timeout=0.5)
        except queue.Empty:
            continue
        _busy.set()
        clip = None
        try:
            t0 = time.time()
            fixture = None
            if mode == "stub":
                fixture = _fixtures.next()
                if fixture is None:
                    print("[vision] fixtures/*.json 이 없어 관측 생략")
                    continue
                time.sleep(STUB_DELAY_SEC)
                obs = analyze_clip(None, mode="stub", fixture=fixture)
            elif mode == "yolo":
                obs = analyze_frames(job["frames"])     # 메모리 프레임만 사용, 파일 저장 없음
            else:
                clip = save_clip(job["frames"], job["fps"])
                obs = analyze_clip(str(clip), mode="local")
            event = make_event(device_id, obs, job, mode, str(fixture) if fixture else None, time.time() - t0)
            print(f"  ✓ 영상 관측 ({mode}, {event['detail']['infer_sec']}s) "
                  f"type={event['type']} msg={event['message'][:50]}")
            notify(event)
        except Exception as e:
            print(f"[vision] 관측 실패: {e}")
        finally:
            if clip is not None:
                clip.unlink(missing_ok=True)   # 영상은 판단에만 쓰고 저장하지 않음
            _busy.clear()


def _submit(job: dict) -> bool:
    if _busy.is_set():
        print("  ↷ 이전 클립 분석 중 → 이번 트리거 생략")
        return False
    try:
        _jobs.put_nowait(job)
        return True
    except queue.Full:
        print("  ↷ 분석 대기열 가득 → 이번 트리거 생략")
        return False


# ───────────────────────── 메인 루프 ─────────────────────────
def _open_capture(source: str):
    import cv2

    cap = cv2.VideoCapture(int(source)) if str(source).isdigit() else cv2.VideoCapture(source)
    if not str(source).isdigit():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 네트워크 스트림 지연 줄이기
    if not cap.isOpened():
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    return cap, (fps if 1e-3 < fps <= 120 else 15.0)


def run(source: str = "0", device_id: str = "homecam-1", mode: str = "yolo",
        preview: bool = False, loop_file: bool = True) -> None:
    """main.py 가 스레드로 실행하는 함수. (preview 창은 단독 실행 때만)"""
    import cv2

    source = str(source)
    try:
        cap, fps = _open_capture(source)
    except Exception as e:
        print(f"[vision] {e} → 영상 감지 중단")
        return
    is_file = not source.isdigit() and Path(source).exists()
    ring: deque[np.ndarray] = deque(maxlen=max(int((CLIP_SEC + POST_SEC) * fps), 1))
    trigger = MotionTrigger(TriggerConfig(min_interval_sec=COOLDOWN_SEC))
    threading.Thread(target=_analysis_worker, args=(device_id, mode), daemon=True).start()
    if mode == "yolo":   # 첫 트리거 때 로딩 지연이 없도록 미리 불러 둠
        threading.Thread(target=load_yolo, daemon=True).start()

    print(f"\n영상 감지 시작 (source={source}, fps≈{fps:.0f}, mode={mode}, "
          f"fixtures={len(list_fixtures())}개)")
    frame_i = 0
    pending: dict | None = None   # 트리거 후 POST_SEC 프레임 모으는 중
    try:
        while not stop.is_set():
            t_loop = time.monotonic()
            ok, frame = cap.read()
            if not ok or frame is None:
                if is_file and loop_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    trigger.reset()
                    continue
                print("[vision] 프레임 없음 → 종료")
                break
            ring.append(frame)
            frame_i += 1
            now = time.time()

            # 1) 트리거 판정: judge 요청 > 모션
            fired, score, reason = False, 0.0, ""
            ext = _pop_check()
            if ext:
                r = trigger.force_fire(now)
                fired, score, reason = True, r.score, ext
            elif frame_i % PROCESS_EVERY_N == 0:
                r = trigger.update(frame, now)
                fired, score, reason = r.fired, r.score, r.reason

            if preview:
                vis = frame.copy()
                cv2.putText(vis, f"motion={score:.3f} {reason}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("vision", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("t"):
                    trigger.force_fire(now)
                    fired, reason = True, "manual"

            if fired and pending is None:
                print(f"[{now_iso()}] ⚡ 영상 트리거 ({reason}, score={score:.3f})")
                job = {"reason": reason, "score": score, "timestamp": now_iso(), "fps": fps}
                if mode == "stub":          # 미리 추론한 결과를 쓰므로 프레임 불필요
                    _submit(job)
                else:
                    pending = {**job, "left": int(POST_SEC * fps)}

            # 2) local: 트리거 이후 POST_SEC 까지 모은 뒤 클립 확정
            if pending is not None:
                pending["left"] -= 1
                if pending["left"] <= 0:
                    _submit({**pending, "frames": list(ring)})
                    pending = None

            if is_file:   # 파일은 실제 속도로 재생
                time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - t_loop)))
    except KeyboardInterrupt:
        print("\n[vision] 사용자 중단")
    finally:
        stop.set()
        cap.release()
        if preview:
            cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description="영상 트리거 → 클립 관측 (단독 실행)")
    ap.add_argument("--source", default="0", help="웹캠 번호 | mp4 경로 | rtsp://...")
    ap.add_argument("--device-id", default="homecam-1")
    ap.add_argument("--mode", choices=("yolo", "stub", "local"), default=os.getenv("VISION_MODE", "yolo"))
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    run(a.source, a.device_id, a.mode, preview=a.preview)


if __name__ == "__main__":
    main()
