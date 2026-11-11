"""
main.py — Flask 웹서버 + 감지 스레드(오디오, 영상)를 한 프로그램으로 실행

  realtime.run  : 마이크/RTSP 오디오 → 위험 소리·도어락 키패드 → judge
  vision.run    : 웹캠/RTSP 영상 → 모션 트리거 → 클립 관측 → judge
  judge         : 소리·영상·키패드를 시간으로 묶어 등급 판정 → MongoDB alerts
  server.app    : 로그인·대시보드, /api/alerts 로 알림 표시

실행 (프로젝트 루트 safe-home/ 에서):  python main.py
환경변수 (.env 에 넣어도 됨)
  DEVICE_ID=homecam-1     소리·영상이 같은 집(device)으로 묶이도록 같은 값 사용
  AUDIO_SOURCE=mic        mic | 파일 경로 | rtsp://...
  VISION_SOURCE=0         웹캠 번호 | 파일 경로 | rtsp://...
  VISION_MODE=stub        stub(미리 추론한 fixture, 시연용) | local(Qwen2.5-VL 실추론)
"""
import os
import threading

from server.app import app            # .env 로드 + DB 연결 확인 (실패하면 여기서 바로 알려 줌)
from detection import realtime, vision

DEVICE_ID = os.getenv("DEVICE_ID", "homecam-1")
AUDIO_SOURCE = os.getenv("AUDIO_SOURCE", "mic")
VISION_SOURCE = os.getenv("VISION_SOURCE", "0")
VISION_MODE = os.getenv("VISION_MODE", "stub")


def start(name, target, **kwargs):
    threading.Thread(target=target, kwargs=kwargs, daemon=True, name=name).start()
    print(f"[main] {name} 시작 {kwargs}")


if __name__ == "__main__":
    start("audio", realtime.run, source=AUDIO_SOURCE, device_id=DEVICE_ID)
    start("vision", vision.run, source=VISION_SOURCE, device_id=DEVICE_ID, mode=VISION_MODE)
    app.run(port=5000, use_reloader=False)  # reloader 켜면 스레드가 2번 뜸
