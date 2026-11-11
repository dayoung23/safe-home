"""
detection/vision.py — 홈캠 영상 실시간 (YOLO 트리거 + 관측 → judge)

  평상시 : 프레임 → YOLOv8n (person) → 상태 게이트. 의미 있는 변화만 이벤트화.
  관측   : mode=yolo → 보수적 매핑 (person / fall / suspicious / none)
           mode=qwen|local → 클립 + backup Qwen
           mode=stub → backup fixtures
  전달   : vision_events.jsonl + risk.judge.on_event
  위험 등급은 정하지 않는다 (judge 담당).

실행
  python main.py
  python -m detection.vision --source 0 --preview
  python -m detection.vision --source demo.mp4 --mode yolo
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
    from detection.yolo_detector import FrameDetections, YoloDetector
    from detection.yolo_trigger import TriggerConfig, YoloTrigger
    from detection.yolo_obs import map_observation
except ImportError:
    from yolo_detector import FrameDetections, YoloDetector
    from yolo_trigger import TriggerConfig, YoloTrigger
    from yolo_obs import map_observation

# ───────────────────────── 설정 ─────────────────────────
CLIP_SEC = float(os.getenv("VISION_CLIP_SEC", "4.0"))
POST_SEC = float(os.getenv("VISION_POST_SEC", "2.0"))
PROCESS_EVERY_N = int(os.getenv("YOLO_EVERY_N", "2"))
COOLDOWN_SEC = float(os.getenv("YOLO_COOLDOWN_SEC", "8.0"))
STUB_DELAY_SEC = float(os.getenv("VISION_STUB_DELAY", "0.4"))
SERVER_URL = os.getenv("VISION_SERVER_URL", "")
FIXED_FIXTURE = os.getenv("VISION_FIXTURE", "")
EVENT_LOG = Path(__file__).resolve().parent / "vision_events.jsonl"
# 웹 대시보드용 카메라 화면: N초마다 박스 그린 프레임 1장을 덮어씀 (0 이면 끔).
# 브라우저는 서버 PC 카메라를 직접 못 여니까, 서버가 이 파일을 갱신하고 웹은 이 이미지를 보여줌.
SNAPSHOT_SEC = float(os.getenv("VISION_SNAPSHOT_SEC", "1.0"))
SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "server" / "static" / "camera" / "live.jpg"


def _write_snapshot(vis) -> None:
    """임시 파일에 쓰고 교체 → 웹이 반쯤 쓴 파일을 읽지 않게."""
    import cv2

    try:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        h, w = vis.shape[:2]
        if w > 640:
            vis = cv2.resize(vis, (640, int(h * 640 / w)))
        tmp = SNAPSHOT_PATH.with_suffix(".tmp.jpg")
        cv2.imwrite(str(tmp), vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        os.replace(tmp, SNAPSHOT_PATH)
    except Exception:
        pass   # 윈도우에서 웹이 읽는 중이면 교체 실패할 수 있음 → 다음 주기에 다시
DEFAULT_MODE = os.getenv("VISION_MODE", "yolo")

stop = threading.Event()
_jobs: "queue.Queue[dict]" = queue.Queue(maxsize=1)
_busy = threading.Event()
_check_lock = threading.Lock()
_check_reason: str | None = None


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def request_check(reason: str = "external") -> None:
    """judge → 다음 주기 강제 관측."""
    global _check_reason
    with _check_lock:
        _check_reason = reason


def _pop_check() -> str | None:
    global _check_reason
    with _check_lock:
        r, _check_reason = _check_reason, None
    return r


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
            try:
                from detection.qwen_backup import list_fixtures
            except ImportError:
                from qwen_backup import list_fixtures
            files = list_fixtures()
            if not files:
                return None
            self._it = itertools.cycle(files)
        return next(self._it)


_fixtures = FixtureCycle()


def make_event(
    device_id: str,
    obs: dict[str, Any],
    job: dict,
    mode: str,
    fixture: str | None,
    infer_sec: float,
) -> dict[str, Any]:
    return {
        "device_id": device_id,
        "event_id": f"vision-{uuid.uuid4().hex[:10]}",
        "source": "vision",
        "type": obs["type"],
        "message": obs["message"],
        "confidence": obs["confidence"],
        "timestamp": now_iso(),
        "scene": obs.get("scene") or "",
        "persons": obs.get("persons") or [],
        "behavior": obs.get("behavior") or "",
        "detail": {
            "trigger": job.get("reason", ""),
            "trigger_time": job.get("timestamp", now_iso()),
            "motion_score": round(float(job.get("score") or 0.0), 4),
            "person_count": int(job.get("person_count") or 0),
            "analyze_mode": mode,
            "fixture": fixture,
            "infer_sec": round(infer_sec, 3),
            "yolo": job.get("yolo"),
        },
    }


def notify(event: dict[str, Any]) -> None:
    with open(EVENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    # judge 는 별도 스레드로 호출 → DB 저장이 느려도 YOLO 프레임 루프는 안 멈춤 (realtime.py 와 동일)
    try:
        from risk import judge
        threading.Thread(target=judge.on_event, args=(event,), daemon=True).start()
    except Exception as e:
        print(f"[vision] judge import 실패: {e}")
    if SERVER_URL:
        try:
            import requests
            requests.post(SERVER_URL, json=event, timeout=3)
        except Exception as e:
            print(f"[vision] 서버 전송 실패: {e}")


def _emit_obs(
    device_id: str,
    obs: dict[str, Any],
    job: dict,
    mode: str,
    fixture: str | None,
    infer_sec: float,
) -> None:
    event = make_event(device_id, obs, job, mode, fixture, infer_sec)
    print(
        f"  ✓ 영상 관측 ({mode}, {event['detail']['infer_sec']}s) "
        f"type={event['type']} msg={event['message'][:60]}"
    )
    notify(event)


def _backup_worker(device_id: str, mode: str) -> None:
    try:
        from detection.qwen_backup import analyze_clip
    except ImportError:
        from qwen_backup import analyze_clip

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
                    print("[vision] backup fixtures 없음 → 관측 생략")
                    continue
                time.sleep(STUB_DELAY_SEC)
                obs = analyze_clip(None, mode="stub", fixture=fixture)
            else:
                frames = job.get("frames") or []
                if not frames:
                    print("[vision] Qwen 백업인데 프레임 없음 → 생략")
                    continue
                clip = save_clip(frames, job["fps"])
                obs = analyze_clip(str(clip), mode="local")
            _emit_obs(
                device_id, obs, job, mode,
                str(fixture) if fixture else None, time.time() - t0,
            )
        except Exception as e:
            print(f"[vision] 백업 관측 실패: {e}")
        finally:
            if clip is not None:
                clip.unlink(missing_ok=True)
            _busy.clear()


def _submit_backup(job: dict) -> bool:
    if _busy.is_set():
        print("  ↷ 이전 백업 분석 중 → 이번 트리거 생략")
        return False
    try:
        _jobs.put_nowait(job)
        return True
    except queue.Full:
        print("  ↷ 백업 대기열 가득 → 이번 트리거 생략")
        return False


def _open_capture(source: str):
    import cv2

    cap = cv2.VideoCapture(int(source)) if str(source).isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    return cap, (fps if 1e-3 < fps <= 120 else 15.0)


def _normalize_mode(mode: str) -> str:
    m = (mode or "yolo").strip().lower()
    if m in ("qwen", "vl"):
        return "local"
    if m in ("yolo", "local", "stub"):
        return m
    print(f"[vision] unknown mode={mode!r} → yolo")
    return "yolo"


def _new_detector(prev: YoloDetector | None = None) -> YoloDetector:
    if prev is None:
        return YoloDetector()
    return YoloDetector(
        model_name=prev.model_name,
        conf=prev.conf,
        iou=prev.iou,
        imgsz=prev.imgsz,
        device=prev.device,
        use_track=prev.use_track,
    )


def _draw_preview(frame, dets: FrameDetections, reason: str, score: float, fall: bool,
                  roi=None, hands_up: bool = False):
    import cv2

    vis = frame.copy()
    h, w = vis.shape[:2]
    if roi:   # 현관 구역 (YOLO_ROI) 표시 → 위치 맞출 때 참고
        x1, y1, x2, y2 = roi
        cv2.rectangle(vis, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (255, 128, 0), 2)
        cv2.putText(vis, "DOOR ZONE", (int(x1 * w) + 6, int(y1 * h) + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
    if hands_up:
        cv2.putText(vis, "HANDS UP", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if fall:
        cv2.putText(vis, "FALL", (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    for p in dets.persons:
        x1, y1, x2, y2 = (int(v) for v in p.xyxy)
        color = (0, 165, 255) if fall else (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis, f"person {p.conf:.2f}", (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
    cv2.putText(
        vis,
        f"yolo n={dets.person_count} {reason} conf={score:.2f}",
        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )
    return vis


def run(
    source: str | None = None,
    device_id: str | None = None,
    mode: str | None = None,
    preview: bool = False,
    loop_file: bool = True,
) -> None:
    """main.py 스레드 진입점. 인자 없으면 VISION_* / DEVICE_ID 환경변수 사용."""
    import cv2

    source = str(source if source is not None else os.getenv("VISION_SOURCE", "0"))
    device_id = str(device_id if device_id is not None else os.getenv("DEVICE_ID", "homecam-1"))
    mode = _normalize_mode(mode if mode is not None else DEFAULT_MODE)
    stop.clear()
    try:
        cap, fps = _open_capture(source)
    except Exception as e:
        print(f"[vision] {e} → 영상 감지 중단")
        return

    is_file = not source.isdigit() and Path(source).exists()
    ring: deque[np.ndarray] = deque(maxlen=max(int((CLIP_SEC + POST_SEC) * fps), 1))
    trigger = YoloTrigger(TriggerConfig(min_interval_sec=COOLDOWN_SEC))
    pending: dict | None = None
    last_dets: FrameDetections | None = None

    detector = _new_detector()
    try:
        detector.load()
    except Exception as e:
        print(f"[vision] YOLO 로드 실패: {e}")
        print("[vision] tip: pip install ultralytics")
        cap.release()
        return

    if mode != "yolo":
        threading.Thread(target=_backup_worker, args=(device_id, mode), daemon=True).start()
        nfix = 0
        try:
            from detection.qwen_backup import list_fixtures
            nfix = len(list_fixtures())
        except Exception:
            pass
        print(f"[vision] obs backup mode={mode} (fixtures={nfix}) — trigger still YOLO")

    print(
        f"\n영상 감지 시작 (source={source}, fps≈{fps:.0f}, mode={mode}, "
        f"every_n={PROCESS_EVERY_N}, cooldown={COOLDOWN_SEC}s)"
    )

    frame_i = 0
    last_snapshot = 0.0
    try:
        while not stop.is_set():
            t_loop = time.monotonic()
            ok, frame = cap.read()
            if not ok or frame is None:
                if is_file and loop_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    trigger.reset()
                    detector = _new_detector(detector)
                    detector.load()
                    last_dets = None
                    continue
                print("[vision] 프레임 없음 → 종료")
                break

            ring.append(frame)
            frame_i += 1
            now = time.time()
            ext = _pop_check()

            run_infer = (frame_i % PROCESS_EVERY_N == 0) or bool(ext)
            if run_infer:
                last_dets = detector.detect(frame)
            dets = last_dets
            if dets is None:
                if is_file:
                    time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - t_loop)))
                continue

            if ext:
                tr = trigger.force_observe(dets, now, reason=ext)
            else:
                tr = trigger.update(dets, now)

            fired, score, reason = tr.fired, tr.score, tr.reason

            if SNAPSHOT_SEC > 0 and now - last_snapshot >= SNAPSHOT_SEC:
                last_snapshot = now
                _write_snapshot(_draw_preview(frame, dets, reason, score, tr.fall_candidate,
                                              roi=trigger.cfg.roi, hands_up=tr.hands_up))

            if preview:
                vis = _draw_preview(frame, dets, reason, score, tr.fall_candidate,
                                    roi=trigger.cfg.roi, hands_up=tr.hands_up)
                cv2.imshow("vision-yolo", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("t"):
                    tr = trigger.force_observe(dets, now, reason="manual")
                    fired, score, reason = True, tr.score, "manual"

            if fired and pending is None:
                print(
                    f"[{now_iso()}] ⚡ 영상 트리거 ({reason}, "
                    f"conf={score:.3f}, n={tr.person_count})"
                )
                job = {
                    "reason": reason,
                    "score": score,
                    "timestamp": now_iso(),
                    "fps": fps,
                    "person_count": tr.person_count,
                    "yolo": dets.as_dict(),
                }
                if mode == "yolo":
                    t0 = time.time()
                    obs = map_observation(dets, tr)
                    _emit_obs(device_id, obs, job, "yolo", None, time.time() - t0)
                elif mode == "stub":
                    _submit_backup(job)
                else:
                    pending = {**job, "left": max(int(POST_SEC * fps), 1)}

            if pending is not None:
                pending["left"] -= 1
                if pending["left"] <= 0:
                    _submit_backup({**pending, "frames": list(ring)})
                    pending = None

            if is_file:
                time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - t_loop)))
    except KeyboardInterrupt:
        print("\n[vision] 사용자 중단")
    finally:
        stop.set()
        cap.release()
        if preview:
            cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO 영상 트리거·관측 (단독 실행)")
    ap.add_argument("--source", default="0", help="웹캠 번호 | mp4 경로 | rtsp://...")
    ap.add_argument("--device-id", default="homecam-1")
    ap.add_argument(
        "--mode",
        choices=("yolo", "stub", "local", "qwen"),
        default=DEFAULT_MODE,
    )
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    # allow re-run in same process
    stop.clear()
    run(a.source, a.device_id, a.mode, preview=a.preview)


if __name__ == "__main__":
    main()
