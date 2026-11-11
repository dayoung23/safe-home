홈 화면 카메라 이미지 폴더

live.jpg     python main.py 실행 중 vision 스레드가 1초마다 자동으로 덮어씀 (YOLO 박스 포함, git 에는 안 올림)
capture.jpg  직접 넣는 캡처 이미지. live.jpg 가 없거나 6초 넘게 갱신이 없으면 이 이미지를 보여줌
             (카메라가 없는 PC 에서 시연할 때, 시연용 장면을 캡처해서 이 이름으로 넣으면 됨)

자동 저장을 끄려면 .env 에 VISION_SNAPSHOT_SEC=0
