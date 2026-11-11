"""
main.py — Flask 웹서버 + 감지 스레드(오디오, 영상)를 한 프로그램으로 실행

  [audio 스레드]  realtime.run : 마이크 → 위험 소리(AST) · 도어락 키패드 → judge
  [vision 스레드] vision.run   : 노트북 캠 → YOLO 상시 탐지 → 상태 트리거 → 관측 → judge
  [judge]         사건 창 5초로 소리·영상·키패드를 묶어 등급 판정 → MongoDB alerts
  [메인 스레드]   Flask :5000  : 로그인·대시보드, /api/alerts (DB 읽기만)

실행 (프로젝트 루트 safe-home/ 에서):  python main.py
환경변수 (.env)
  DEVICE_ID=homecam-1   소리·영상이 같은 집으로 묶이도록 같은 값
  AUDIO_SOURCE=mic      mic | 파일 경로 | rtsp://...
  VISION_SOURCE=0       노트북 캠 번호 | 파일 경로 | rtsp:// | http://.../stream.mjpg
  VISION_MODE=yolo      yolo (기본) | stub | local(qwen)
"""
import os
import threading

from server.app import app            # .env 로드 + DB 연결 확인
from detection import realtime, vision

DEVICE_ID = os.getenv("DEVICE_ID", "homecam-1")
AUDIO_SOURCE = os.getenv("AUDIO_SOURCE", "mic")
VISION_SOURCE = os.getenv("VISION_SOURCE", "0")
VISION_MODE = os.getenv("VISION_MODE", "yolo")


def start(name, target, **kwargs):
    threading.Thread(target=target, kwargs=kwargs, daemon=True, name=name).start()
    print(f"[main] {name} 시작 {kwargs}")


if __name__ == "__main__":
    start("audio", realtime.run, source=AUDIO_SOURCE, device_id=DEVICE_ID)
    start("vision", vision.run, source=VISION_SOURCE, device_id=DEVICE_ID, mode=VISION_MODE)
    app.run(port=5000, use_reloader=False)  # reloader 켜면 스레드가 2번 뜸
