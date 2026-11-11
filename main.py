"""
main.py — 감지 스레드 시작 + Flask 서버 실행. 프로젝트 루트(safe-home/)에서 python main.py
"""
import threading

from detection import realtime, vision      # realtime = 소리 + 키패드 (sound/keypad는 realtime이 내부에서 씀)
from server.app import app


def start_thread(module):
    if hasattr(module, "run"):
        threading.Thread(target=module.run, daemon=True).start()
        print(f"[main] {module.__name__} 시작")
    else:
        print(f"[main] {module.__name__}.run 없음 → 건너뜀 (아직 머지 전)")


if __name__ == "__main__":
    start_thread(realtime)
    start_thread(vision)
    app.run(port=5000, use_reloader=False)  # reloader 켜면 스레드가 2번 뜸