import cv2
import numpy as np
import threading
import queue
import socket
import requests

from actiontools.tools import *

# YOLO Pose 모델 로드
from ultralytics import YOLO
model = YOLO("./models/yolo11n-pose.pt")  # 원하는 모델을 다운로드 후 경로 수정 가능


def extract_keypoints(results):
    results = results.reshape(17,3)
    pose = np.array([[res[0], res[1], res[2]] for idx, res in enumerate(results)]).flatten()
    return pose


# Socket & streaming setup
ROBOT_IP = "192.168.0.123"
ROBOT_PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)

frame_queue = queue.Queue(maxsize=5)
running = True
cur_action = 'nothing'
frame_counter = 1
start_time = None
swing_analysis = True



# 스트리밍 URL 및 상태 변수
url = "http://192.168.0.129:5000/video_front"

# 로컬 웹캠에서 프레임 가져오기
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


# 실시간 프레임 처리 함수
def process_frames():
    global running, cur_action, frame_counter, start_time, swing_analysis
    print("프레임 처리 시작...")
    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()

            
            try:
                results = model(frame, verbose=False)

                # 결과에서 keypoints 가져오기
                for result in results:
                    jointsdata = result.keypoints.data.cpu().numpy() 
                    keypoints, scores = jointsdata[:1,:,:2], jointsdata[:1,:,2:]
                    if scores.any():
                        h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(keypoints, scores)
                        kps = np.concatenate([h36m_kpts, h36m_scores], axis=2)
                        frame = show2Dpose(h36m_kpts, frame)
                        # frame = show2Dpose_h36m(frame, kps)
                

            except:
                print('something error')
            
            cv2.imshow("Action", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                running = False

# 스레드 실행
capture_thread = threading.Thread(target=stream_frames)
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()
capture_thread.join()
process_thread.join()

cv2.destroyAllWindows()
print("프로그램 종료.")
