"""
pi/camera_stream.py — 라즈베리파이를 홈캠으로: 카메라 영상을 MJPEG 로 스트리밍

파이는 영상만 보내고, 판단(모션 트리거·YOLO·judge)은 노트북의 safe-home 이 한다.
  파이:   python3 camera_stream.py                 → http://<파이IP>:8000/stream.mjpg
  노트북: VISION_SOURCE=http://<파이IP>:8000/stream.mjpg python main.py

카메라
  - 파이 카메라 모듈(CSI): picamera2 사용 (Raspberry Pi OS 에 기본 설치)
  - USB 웹캠: picamera2 가 없거나 --usb 옵션이면 OpenCV 로 /dev/video0 사용
의존성: 파이 카메라 → 추가 설치 없음 / USB → sudo apt install python3-opencv
"""
import argparse
import io
import socketserver
import threading
import time
from http import server

PAGE = b"""<html><body style="margin:0;background:#000">
<img src="/stream.mjpg" style="width:100%"></body></html>"""


class FrameBuffer(io.BufferedIOBase):
    """최신 JPEG 1장만 보관. 클라이언트들은 새 프레임이 올 때까지 기다렸다 받음."""

    def __init__(self):
        self.frame = None
        self.cond = threading.Condition()

    def write(self, buf):
        with self.cond:
            self.frame = bytes(buf)
            self.cond.notify_all()
        return len(buf)


def start_picamera(buf: FrameBuffer, w: int, h: int, fps: int):
    from picamera2 import Picamera2
    from picamera2.encoders import JpegEncoder
    from picamera2.outputs import FileOutput

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (w, h)}, controls={"FrameRate": fps}))
    cam.start_recording(JpegEncoder(q=70), FileOutput(buf))
    print(f"[pi] 파이 카메라 모듈 시작 {w}x{h}@{fps}")
    return cam


def start_usb(buf: FrameBuffer, w: int, h: int, fps: int, index: int = 0):
    import cv2

    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise RuntimeError(f"USB 카메라 {index} 를 열 수 없습니다")

    def loop():
        delay = 1.0 / fps
        while True:
            ok, frame = cap.read()
            if ok:
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    buf.write(jpg.tobytes())
            time.sleep(delay)

    threading.Thread(target=loop, daemon=True).start()
    print(f"[pi] USB 카메라 {index} 시작 {w}x{h}@{fps}")
    return cap


def make_handler(buf: FrameBuffer):
    class Handler(server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                self.end_headers()
                try:
                    while True:
                        with buf.cond:
                            buf.cond.wait()
                            frame = buf.frame
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                except Exception:
                    pass   # 클라이언트 연결 끊김
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass

    return Handler


class Server(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--usb", action="store_true", help="파이 카메라 대신 USB 웹캠 사용")
    a = ap.parse_args()

    buf = FrameBuffer()
    if a.usb:
        start_usb(buf, a.width, a.height, a.fps)
    else:
        try:
            start_picamera(buf, a.width, a.height, a.fps)
        except Exception as e:
            print(f"[pi] 파이 카메라 실패({e}) → USB 카메라 시도")
            start_usb(buf, a.width, a.height, a.fps)

    print(f"[pi] 스트리밍: http://<이 파이의 IP>:{a.port}/stream.mjpg  (IP 확인: hostname -I)")
    Server(("", a.port), make_handler(buf)).serve_forever()


if __name__ == "__main__":
    main()
