"""
main.py — 감지 스레드 시작 + Flask 서버. 프로젝트 루트에서 python main.py

  vision : YOLO 트리거·관측 (VISION_SOURCE / VISION_MODE)
  audio  : detection.realtime (있으면). 없으면 영상만.

환경변수: DEVICE_ID, AUDIO_SOURCE, VISION_SOURCE, VISION_MODE=yolo
"""
import os
import sys
import threading

from server.app import app
from detection import vision

DEVICE_ID = os.getenv("DEVICE_ID", "homecam-1")
AUDIO_SOURCE = os.getenv("AUDIO_SOURCE", "mic")
VISION_SOURCE = os.getenv("VISION_SOURCE", "0")
VISION_MODE = os.getenv("VISION_MODE", "yolo")


def start(name, target, **kwargs):
    threading.Thread(target=target, kwargs=kwargs, daemon=True, name=name).start()
    print(f"[main] {name} 시작 {kwargs}")


def _audio_main():
    """seungwoo realtime 은 main()+argparse 라서 argv 를 맞춰 호출."""
    try:
        from detection import realtime
    except Exception as e:
        print(f"[main] realtime import 실패 → 오디오 생략: {e}")
        return
    sys.argv = ["realtime", "--source", AUDIO_SOURCE, "--device-id", DEVICE_ID]
    realtime.main()


if __name__ == "__main__":
    start("audio", _audio_main)
    start(
        "vision",
        vision.run,
        source=VISION_SOURCE,
        device_id=DEVICE_ID,
        mode=VISION_MODE,
    )
    app.run(port=5000, use_reloader=False)
