import cv2
import numpy as np
import threading
import queue
import socket
import requests
import torch 
from actiontools.torchNet import LSTMWithResBlock 

from actiontools.tools import *


# 스트리밍 URL 및 상태 변수
url = "http://192.168.0.129:5000/video_front"
# Socket & streaming setup
ROBOT_IP = "192.168.0.123"
ROBOT_PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)


# ######################### YOLO Pose 모델 로드 #########################################
from ultralytics import YOLO
pose_model = YOLO("./models/yolo11n-pose.pt")  # 원하는 모델을 다운로드 후 경로 수정 가능


def extract_keypoints(results):
    results = results.reshape(17,3)
    pose = np.array([[res[0], res[1], res[2]] for idx, res in enumerate(results)]).flatten()
    return pose

# #################################### Action Recogniton 모델 로드 ###################################

# Actions 설정
actions = np.array(['nothing', 'ready', 'stop', 'emergency'])  # 구분할 동작
num_classes = len(actions)
sequence_length = 15  # 프레임 길이
input_size = 51
threshold = 0.8      # 예측 임계값 
hidden_size = 128

# LSTM 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMWithResBlock(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("./models/action/best_kpcl.pth", map_location=device))
model.eval()

#########################################################################

frame_queue = queue.Queue(maxsize=5)
running = True
cur_action = 'nothing'
frame_counter = 1
start_time = None
swing_analysis = True

# 변수 초기화
sequence = []  # 15 프레임씩 쌓을 리스트
sentence = []  # 예측된 동작 저장 리스트

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
    global running, cur_action, frame_counter, start_time, swing_analysis, sequence, sentence
    print("프레임 처리 시작...")
    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_size = frame.shape
            try:
                results = pose_model(frame, verbose=False)

                # 결과에서 keypoints 가져오기
                for result in results:
                    jointsdata = result.keypoints.data.cpu().numpy() 
                    keypoints, scores = jointsdata[:1,:,:2], jointsdata[:1,:,2:]
                    if scores.any():
                        h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(keypoints, scores)
                        frame = show2Dpose(h36m_kpts, frame)
                        norm2dhp = normalize2dhp(h36m_kpts, w=frame_size[1], h=frame_size[0])
                        keypoints = extract_keypoints(np.concatenate([norm2dhp, h36m_scores], axis=2))

                        # 시퀀스에 키포인트 추가
                        sequence.append(keypoints)
                        sequence = sequence[-sequence_length:]
                        
                        
                        if len(sequence) == sequence_length:
                            try:
                                input_seq = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 15, 34)
                                predictions = model(input_seq)
                                res = predictions.detach().cpu().numpy()[0]
                                select = np.argmax(res)
                            except:
                                sequence = []
                                continue
                            
                            
                            #3. Viz logic
                            if res[select] > threshold: 
                                if len(sentence) > 0: 
                                    if actions[select] != sentence[-1]:
                                        sentence.append(actions[select])
                                else:
                                    sentence.append(actions[select])
                            
                            else:
                                sentence = []
                            

                            if len(sentence) > 1: 
                                sentence = sentence[-1:]

                            # Viz probabilities
                            frame = prob_viz(res, actions, frame)

            except:
                print('something error')
            
            cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
     
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
