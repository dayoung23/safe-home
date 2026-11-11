"""홈캠 오디오 실시간 감지 (AST 기반) - 위험 소리 + 도어락 키패드 통합"""
import argparse
import datetime
import json
import os
import subprocess
import threading
import time
from collections import deque

import numpy as np
import requests

from sound import classifier, WATCH_THRESHOLDS
from keypad import analyze_keypad_audio, LOCK_TOP_LABELS

# ───────────────────────── 설정 ─────────────────────────
SR = 16000
CHUNK_SEC = 0.25
WINDOW_SEC = 4.0
HOP_SEC = 2.0
BUFFER_SEC = 60.0
SILENCE_RMS = 0.005
COOLDOWN_SEC = 30
SERVER_URL = ""
EVENT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realtime_events.jsonl")

KEYPAD_PRE_ROLL_SEC = 0.8
KEYPAD_QUIET_SEC = 3.0
KEYPAD_MAX_SESSION_SEC = 30.0
KEYPAD_REFRACTORY_SEC = 4.0
LOCK_COOLDOWN_SEC = 10.0
KEYPAD_FAIL_DANGER = 3
KEYPAD_FAIL_WINDOW_SEC = 180.0

RISK_RANK = {
    "Screaming": 2, "Glass": 2, "Shatter": 2, "Smash, crash": 2, "Breaking": 2,
    "Shout": 1, "Yell": 1, "Slam": 1, "Bang": 1, "Thump, thud": 1, "Crying, sobbing": 1,
}
LEVEL_NAME = {0: "normal", 1: "warning", 2: "danger"}
LEVEL_KO = {1: "주의", 2: "위험"}
LABEL_KO = {
    "Screaming": "비명", "Shout": "고함", "Yell": "고함", "Slam": "문 쾅",
    "Bang": "쾅 소리", "Thump, thud": "둔탁한 충격음", "Smash, crash": "부서지는 소리",
    "Glass": "유리 소리", "Shatter": "유리 깨지는 소리", "Breaking": "파손 소리",
    "Crying, sobbing": "울음소리",
}

stop = threading.Event()
finished = threading.Event()


class RingBuffer:
    def __init__(self, seconds: float):
        self.size = int(seconds * SR)
        self.buf = np.zeros(self.size, dtype=np.float32)
        self.total = 0
        self.lock = threading.Lock()

    def push(self, x: np.ndarray):
        with self.lock:
            n = len(x)
            if n >= self.size:
                self.buf[:] = x[-self.size:]
            else:
                self.buf = np.concatenate([self.buf[n:], x])
            self.total += n

    def last(self, n: int):
        with self.lock:
            n = min(n, self.size)
            return self.buf[-n:].copy(), self.total

    def range(self, start: int, end: int) -> np.ndarray:
        with self.lock:
            s = max(start, self.total - self.size, 0)
            e = min(end, self.total)
            if e <= s:
                return np.zeros(0, dtype=np.float32)
            i_s = self.size - (self.total - s)
            i_e = self.size - (self.total - e)
            return self.buf[i_s:i_e].copy()


def capture_mic(ring: RingBuffer, device):
    import sounddevice as sd

    def callback(indata, frames, t, status):
        ring.push(indata[:, 0].copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=int(CHUNK_SEC * SR), device=device, callback=callback):
        while not stop.is_set():
            time.sleep(0.5)


def capture_ffmpeg(ring: RingBuffer, source: str, is_file: bool):
    cmd = ["ffmpeg", "-loglevel", "error"]
    if is_file:
        cmd += ["-re"]
    elif source.startswith("rtsp"):
        cmd += ["-rtsp_transport", "tcp"]
    cmd += ["-i", source, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    chunk_bytes = int(CHUNK_SEC * SR) * 2

    while not stop.is_set():
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        while not stop.is_set():
            raw = proc.stdout.read(chunk_bytes)
            if not raw:
                break
            raw = raw[: len(raw) // 2 * 2]
            ring.push(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
        proc.kill()

        if is_file:
            finished.set()
            return
        print("[경고] 스트림이 끊겼습니다. 3초 후 재연결합니다...")
        time.sleep(3)


def rms_of(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def evaluate(window: np.ndarray):
    t0 = time.time()
    preds = classifier({"raw": window, "sampling_rate": SR}, top_k=527)
    infer_sec = time.time() - t0

    scores = {p["label"]: float(p["score"]) for p in preds}
    watch = {k: round(scores[k], 4) for k in WATCH_THRESHOLDS if k in scores}
    watch = dict(sorted(watch.items(), key=lambda x: -x[1]))

    alerts = [k for k, v in watch.items()
              if v >= WATCH_THRESHOLDS[k] and RISK_RANK.get(k, 0) > 0]
    level = max((RISK_RANK[k] for k in alerts), default=0)
    if len(alerts) >= 2:
        level = 2
    return preds[:3], watch, alerts, level, infer_sec


def keypad_risk(res: dict, fail_times: deque) -> str:
    # 1. 연속 오류음 3회 발생 시 danger
    if res["type"] == "keypad_failed":
        now = time.time()
        fail_times.append(now)
        while fail_times and now - fail_times[0] > KEYPAD_FAIL_WINDOW_SEC:
            fail_times.popleft()
        return "danger" if len(fail_times) >= KEYPAD_FAIL_DANGER else "warning"

    # 2. 리듬 불일치(keypad_anomaly)이거나 느린 입력은 warning
    if res["type"] in ["keypad_slow_input", "keypad_anomaly"]:
        return "warning"

    return "normal"


def make_hazard_event(device_id, ts, primary, level, alerts, watch, top3) -> dict:
    return {
        "device_id": device_id,
        "source": "audio",
        "type": primary,
        "message": f"{LABEL_KO.get(primary, primary)} 소리 감지 - {LEVEL_KO[level]}",
        "confidence": watch[primary],
        "risk_level": LEVEL_NAME[level],
        "timestamp": ts,
        "top_3": [{"label": p["label"], "score": round(float(p["score"]), 4)} for p in top3],
        "detail": {"alerts": alerts, "watch": dict(list(watch.items())[:6])},
    }


def make_keypad_event(res: dict, device_id: str, risk: str) -> dict:
    # verification 필드를 detail에 포함
    keys = ("auth_status", "typing_speed", "beep_count", "avg_interval_sec", "beep_intervals", "verification", "confidence_breakdown")
    return {
        "device_id": device_id,
        "source": "keypad",
        "type": res["type"],
        "message": res["message"],
        "confidence": res["confidence"],
        "risk_level": risk,
        "timestamp": res["timestamp"],
        "top_3": res["top_3"],
        "detail": {k: res[k] for k in keys if k in res},
    }


def notify(event: dict):
    with open(EVENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    if SERVER_URL:
        try:
            requests.post(SERVER_URL, json=event, timeout=3)
        except Exception as e:
            print(f"[경고] 서버 전송 실패: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="mic", help="mic | 파일 경로 | rtsp://...")
    parser.add_argument("--device-id", default="homecam-1")
    parser.add_argument("--mic-device", type=int, default=None)
    args = parser.parse_args()

    ring = RingBuffer(BUFFER_SEC)
    is_file = os.path.exists(args.source)

    if args.source == "mic":
        target, t_args = capture_mic, (ring, args.mic_device)
    else:
        target, t_args = capture_ffmpeg, (ring, args.source, is_file)
    threading.Thread(target=target, args=t_args, daemon=True).start()

    win_n = int(WINDOW_SEC * SR)
    quiet_n = int(KEYPAD_QUIET_SEC * SR)

    print(f"\n실시간 감지 시작 (source={args.source}, 창 {WINDOW_SEC}s / 주기 {HOP_SEC}s). 종료: Ctrl+C")
    print("=" * 75)

    last_sent = {"t": 0.0, "level": 0}
    session = None
    refractory_until = 0
    last_lock_t = 0.0
    fail_times = deque()

    try:
        while not stop.is_set():
            cycle_start = time.monotonic()
            window, total = ring.last(win_n)

            if total >= win_n or finished.is_set():
                ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                rms = rms_of(window)
                top3 = []

                if rms < SILENCE_RMS:
                    print(f"[{ts}] rms={rms:.4f} 무음 - 분석 생략")
                else:
                    top3, watch, alerts, level, infer = evaluate(window)
                    top1 = top3[0]
                    print(f"[{ts}] rms={rms:.3f} top1={top1['label']}({top1['score']:.2f}) "
                          f"risk={LEVEL_NAME[level]} (추론 {infer:.1f}s)")

                    if level > 0:
                        now_t = time.time()
                        cooled = now_t - last_sent["t"] >= COOLDOWN_SEC
                        escalated = level > last_sent["level"]
                        if cooled or escalated:
                            primary = max(alerts, key=lambda k: (RISK_RANK[k], watch[k]))
                            event = make_hazard_event(args.device_id, ts, primary, level, alerts, watch, top3)
                            notify(event)
                            last_sent = {"t": now_t, "level": level}
                            print(f"  ⚠ 알림 전송: {event['message']} (alerts={alerts})")

                top1_label = top3[0]["label"] if top3 else None

                if session is None:
                    if top1_label == "Beep, bleep" and total >= refractory_until:
                        start = max(0, total - int(KEYPAD_PRE_ROLL_SEC * SR))
                        session = {"start": start}
                        print("  🔐 키패드 소리 감지 - 입력 완료까지 수집 중...")
                    elif top1_label in LOCK_TOP_LABELS and time.time() - last_lock_t >= LOCK_COOLDOWN_SEC:
                        res = analyze_keypad_audio(window, SR, top3)
                        event = make_keypad_event(res, args.device_id, keypad_risk(res, fail_times))
                        notify(event)
                        last_lock_t = time.time()
                        print(f"  🔐 도어락 이벤트: {event['message']} (risk={event['risk_level']})")

                elif session is not None:
                    quiet_rms = rms_of(ring.last(quiet_n)[0])
                    elapsed = (total - session["start"]) / SR
                    if quiet_rms < SILENCE_RMS or elapsed >= KEYPAD_MAX_SESSION_SEC or finished.is_set():
                        audio = ring.range(session["start"], total)
                        preds = classifier({"raw": audio, "sampling_rate": SR}, top_k=527)[:3]
                        res = analyze_keypad_audio(audio, SR, preds)
                        event = make_keypad_event(res, args.device_id, keypad_risk(res, fail_times))
                        notify(event)

                        # 실시간 판정 콘솔 출력에 유사도 및 인증 결과 표기
                        v = res.get("verification", {})
                        sim = f"{v.get('similarity_pct', 0)}%" if 'similarity_pct' in v else "-"
                        verdict = "PASS" if v.get("is_owner") else "FAIL"
                        print(f"  🔐 키패드 판정: {event['message']} | 인증: {verdict} ({sim}) | risk={event['risk_level']}")

                        session = None
                        refractory_until = total + int(KEYPAD_REFRACTORY_SEC * SR)

            if finished.is_set():
                print("\n[종료] 파일 재생이 끝났습니다.")
                break
            time.sleep(max(0.0, HOP_SEC - (time.monotonic() - cycle_start)))
    except KeyboardInterrupt:
        print("\n[종료] 사용자 중단")
    finally:
        stop.set()


if __name__ == "__main__":
    main()