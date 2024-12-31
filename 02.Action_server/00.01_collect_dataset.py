import cv2
import os 
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


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Dataset') 

# Actions that we try to detect
actions = np.array(['nothing, ready, stop, emergency'])   # nothing, ready, stop, emergency  

# Thirty videos worth of data
no_sequences = 100
# Videos are going to be 30 frames in length
sequence_length = 15

start = 0
for action in actions: 
    for sequence in range(start, start+no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
        
        
# 스트리밍 URL 및 상태 변수
url = "http://192.168.0.129:5000/video_front"

frame_queue = queue.Queue(maxsize=5)
# 로컬 웹캠에서 프레임 가져오기
def stream_frames():
    print("스트리밍 시작...")
    stream = requests.get(url, stream=True, timeout=5)
    if stream.status_code == 200:
        byte_data = b""
        for chunk in stream.iter_content(chunk_size=1024):
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



# NEW LOOP
# Loop through actions
def process_frames():
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(start, start+no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                frame = frame_queue.get()
                frame_size = frame.shape
                
                # try:
                results = model(frame, verbose=False)

                # 결과에서 keypoints 가져오기
                for result in results:
                    jointsdata = result.keypoints.data.cpu().numpy() 
                    keypoints, scores = jointsdata[:1,:,:2], jointsdata[:1,:,2:]
                    if scores.any():
                        h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(keypoints, scores)
                        kps = np.concatenate([h36m_kpts, h36m_scores], axis=2)
                        frame = show2Dpose(h36m_kpts, frame)
                
            
                if scores.any():
                    # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(frame, f'STARTING COLLECTION : {action}', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(frame, f'Video Number {sequence}/{start+no_sequences}', (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 6, cv2.LINE_AA)
                        # Show to screen
                        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
                        cv2.imshow('OpenCV Feed', frame)
                        cv2.waitKey(100)
                    else: 
                        cv2.putText(frame, f'Video Number {sequence}/{start+no_sequences}', (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 6, cv2.LINE_AA)
                        # Show to screen
                        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
                        cv2.imshow('OpenCV Feed', frame)
                    
                    # 정규화 
                    norm2dhp = normalize2dhp(h36m_kpts, w=frame_size[1], h=frame_size[0])
                    norm2dhpc = np.concatenate([norm2dhp, h36m_scores], axis=2)
                    # NEW Export keypoints
                    keypoints = extract_keypoints(norm2dhpc)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                
                # except:
                #     print('something error')

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    
    cv2.destroyAllWindows()
    print("프로그램 종료.")         

# 스레드 실행
capture_thread = threading.Thread(target=stream_frames)
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()
capture_thread.join()
process_thread.join()

cv2.destroyAllWindows()
print("프로그램 종료.")