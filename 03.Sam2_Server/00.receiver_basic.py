import cv2
import requests
import numpy as np
import threading
import queue
import torch
import socket


# 모델 및 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# 스트리밍 URL 및 상태 변수
url = "http://192.168.0.129:5000/video_front"

# 모바일 로봇의 IP 주소와 포트 설정
ROBOT_IP = "192.168.0.123"  # 모바일 로봇의 IP 주소 (변경 필요)
ROBOT_PORT = 5000  # 모바일 로봇의 수신 포트


# 소켓 생성
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)  # 송신 버퍼 크기 설정


frame_queue = queue.Queue(maxsize=5)  # 프레임 큐
running = True


# 실시간 스트림을 수신하는 스레드
def stream_frames():
    global running
    print("스트리밍 시작...")
    stream = requests.get(url, stream=True, timeout=5)
    if stream.status_code == 200:
        byte_data = b""
        for chunk in stream.iter_content(chunk_size=1024):
            if not running:
                break
            byte_data += chunk
            a = byte_data.find(b'\xff\xd8')
            b = byte_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = byte_data[a:b+2]
                byte_data = byte_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                frame = cv2.flip(frame, 1)
                if not frame_queue.full():
                    frame_queue.put(frame)
    else:
        print(f"스트리밍 실패: {stream.status_code}")



# YOLO 및 SAM2로 프레임 처리하는 스레드
def process_frames():
    global running
    print("프레임 처리 시작...")
    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # 최종 프레임 출력
            cv2.imshow("YOLO Object Detection & SAM2 Segmentation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break


# 스레드 시작
stream_thread = threading.Thread(target=stream_frames)
process_thread = threading.Thread(target=process_frames)

stream_thread.start()
process_thread.start()


# 스레드 종료 대기
stream_thread.join()
process_thread.join()


# 리소스 정리
cv2.destroyAllWindows()
print("프로그램 종료.")
