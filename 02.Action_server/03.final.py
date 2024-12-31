import cv2
import numpy as np
import threading
import queue
import socket
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
import shutil
import requests
from datetime import datetime

from actiontools.torchNet import LSTMWithResBlock 
from actiontools.tools import *
from actiontools.robot import control_robot

from dataloaders.datapreprocessing import SampleVideo
from networks.model1_1 import EventDetector
from dataloaders.dataloader_with_pose1 import ToTensor, Normalize
from utils.getposes import get_pose3D, get_pose2D_yolov11
from lib.hrnet.gen_kpts import get_frame_images


# 스트리밍 URL 및 상태 변수
url = "http://192.168.0.129:5000/video_front"
# http://192.168.0.123:5000/video_back
# Socket & streaming setup
ROBOT_IP = "192.168.0.126"
ROBOT_PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)

# ################################## 2D HPE 모델 세팅 ######################
from ultralytics import YOLO
pose_model = YOLO("./models/yolo11n-pose.pt")  # 원하는 모델을 다운로드 후 경로 수정 가능

# Keypoints 추출 함수
def extract_keypoints(results):
    results = results.reshape(17, 3)  # (17, 3) -> x, y, confidence
    pose = np.array([[res[0], res[1], res[2]] for res in results]).flatten()  
    return pose

# ################################## Action recognition 모델 세팅 ############# 
threshold = 0.7
actions = np.array(['nothing', 'ready', 'stop', 'emergency']) 
num_classes = len(actions)
sequence_length = 15
input_size = 51
hidden_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 불러오기
model_action = LSTMWithResBlock(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)
model_action.load_state_dict(torch.load('./models/action/best_kpcl.pth', map_location=device))
model_action.to(device)
model_action.eval()

# ################################# Event Detection 모델 세팅 ################

# Event Detection setup
seq_length = 64
event_names = {0: 'Address', 1: 'Toe-up', 2: 'Backswing', 3: 'Top', 4: 'Downswing', 5: 'Impact', 6: 'Follow-through', 7: 'Finish'}
event_model_path = 'models/pose1_2'
frames_folder = './frames'
os.makedirs(frames_folder, exist_ok=True)
shutil.rmtree(frames_folder)
output_dir = './demo/output'
os.makedirs(frames_folder, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Event Detector 모델 로드
model_event = EventDetector(pretrain=True, width_mult=1., lstm_layers=1, lstm_hidden=256, bidirectional=True, dropout=False)
load_model_name = sorted(glob.glob(f"{event_model_path}/*.tar"))[-1]
model_event.load_state_dict(torch.load(load_model_name)['model_state_dict'])
model_event.to(device)  # GPU로 이동
model_event.eval()

##########################################################################

frame_queue = queue.Queue(maxsize=5)
running = True
cur_action = 'nothing'
frame_counter = 1
start_time = None
swing_analysis = True
swing_analysis_ing = False
swing_analysis_kpts_result = []
swing_analysis_scores_result = []

sequence = [] 
sentence = []
frames = []

# 로컬 웹캠에서 프레임 가져오기
def capture_frames():
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


# Event Detection 및 HPE 처리 스레드
def event_detection_hpe():
    global model_event,frames_folder, output_dir, swing_analysis, swing_analysis_ing, cur_action
    global swing_analysis_kpts_result, swing_analysis_scores_result, frame_size, frames
    
    print("Event Detection 및 HPE 시작...")
    swing_analysis_ing = True 
    
    keypoints = np.array(swing_analysis_kpts_result)
    scores = np.array(swing_analysis_scores_result)
    kps_conf = np.concatenate([keypoints, scores], axis=3)
    poses = kps_conf.transpose(1, 0, 2, 3)[0]
    width, height, _ = frame_size

    # poses, width, height, frames = get_pose2D_yolov11(frames_folder)

    ds = SampleVideo(poses, width, height, mode=5, transform=transforms.Compose([
        ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    # Event Detection 수행
    for sample in dl:
        inputs = sample['inputs']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < inputs.shape[1]:
            if (batch + 1) * seq_length > inputs.shape[1]:
                input_batch = inputs[:, batch * seq_length:, :]
            else:
                input_batch = inputs[:, batch * seq_length:(batch + 1) * seq_length, :]
            logits = model_event(input_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    # 결과 저장
    events = np.argmax(probs, axis=0)[:-1]
    print(f"Predicted event frames: {events}")
    image_paths = get_frame_images(frames_folder)

    current_time = datetime.now().strftime("%Y%m%d%H%M")  # 년월일시분
    save_folder = f'{output_dir}/{current_time}/'
    os.makedirs(save_folder, exist_ok=True)


    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print('Condifence: {}'.format([np.round(c, 3) for c in confidence]))
    
    thickness=1
    loc1=(20, 20)
    
    event_save = save_folder + 'event/'
    os.makedirs(event_save, exist_ok=True)
    for i, e in enumerate(events):
        if e < len(image_paths):
            # img = image_paths[e]
            img = cv2.imread(image_paths[e])
            if img.shape[0] > 1000:
                cv2.namedWindow(event_names[i], cv2.WINDOW_NORMAL)
                cv2.resizeWindow(event_names[i], int(img.shape[0] * 0.8), int(img.shape[0] * 0.8))

            cv2.putText(img, '{}'.format(event_names[i]), loc1, cv2.FONT_HERSHEY_DUPLEX, thickness, (0, 0, 255))
            cv2.imwrite(event_save + "{}.{}.jpg".format(i, event_names[i]), img)
            
    print('Generating demo successful!')

    # 2D & 3D HPE 저장
    get_pose3D(poses, width, height, '27_243_45.2.bin', frames, save_folder)
    print("Event Detection 및 HPE 저장 완료!")
    swing_analysis = True 
    swing_analysis_ing = False
    cur_action = 'nothing'
    os.makedirs(frames_folder, exist_ok=True)
    shutil.rmtree(frames_folder)
    os.makedirs(frames_folder, exist_ok=True)
    swing_analysis_kpts_result, swing_analysis_scores_result = [], []
    frames = []

# 실시간 프레임 처리 함수
def process_frames():
    global running, sequence, cur_action, frame_counter, start_time, swing_analysis, swing_analysis_ing, sentence
    global swing_analysis_kpts_result, swing_analysis_scores_result, frame_size, frames
    print("프레임 처리 시작...")
    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()
            raw_frame = frame.copy()
            frame_size = frame.shape
            results = pose_model(frame, verbose=False)
            if len(results[0].boxes.data.cpu().numpy()) == 0:
                control_robot(sock, ROBOT_IP, ROBOT_PORT, 'no person')
            
            for result in results:
                jointsdata = result.keypoints.data.cpu().numpy()  # (1, 17, 3)
                keypoints, scores = jointsdata[:1, :, :2], jointsdata[:1, :, 2:]

                if scores.any():
                    # 키포인트 처리 및 정규화
                    h36m_kpts, h36m_scores, _ = h36m_coco_format(keypoints, scores)
                    frame = show2Dpose(h36m_kpts, frame)  # 2D Pose 그리기
                    frame_hpe = frame.copy()
                    norm2dhp = normalize2dhp(h36m_kpts, w=frame_size[1], h=frame_size[0])
                    keypoints = extract_keypoints(np.concatenate([norm2dhp, h36m_scores], axis=2))
                    sequence.append(keypoints)
                    sequence = sequence[-sequence_length:]

                else:
                    sequence = []
                    cur_action = 'nothing'
            
                    # Action Recognition
                if len(sequence) == sequence_length and swing_analysis:
                    try:
                        with torch.no_grad():
                            input_data = torch.tensor([sequence], dtype=torch.float32).to(device)
                            predictions = model_action(input_data)
                            probabilities = torch.softmax(predictions, dim=1)[0]  # 확률 분포
                            action_index = torch.argmax(probabilities).item()
                            action_confidence = probabilities[action_index].item()
                            cur_action = actions[action_index]
                            # print(f"Predicted Action: {actions[action_index]}, Confidence: {action_confidence:.2f}")
                    except:
                        sequence = []
                        continue
                    
                    # control_robot(sock, ROBOT_IP, ROBOT_PORT, cur_action)
                    
                    # 예측 결과가 임계값을 넘으면 sentence에 추가
                    if action_confidence+0.3 > threshold:
                        control_robot(sock, ROBOT_IP, ROBOT_PORT, cur_action)
                        if len(sentence) == 0 or actions[action_index] != sentence[-1]:
                            sentence.append(actions[action_index])
                    else:
                        sentence = []

                    if len(sentence) > 1:
                        sentence = sentence[-1:]
                        cv2.putText(frame, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    frame = prob_viz(probabilities.cpu().numpy()+0.3, actions, frame)

                # 'ready' 상태에서 이미지 저장
                if cur_action == 'ready' and not swing_analysis_ing:
                    if start_time is None:
                        control_robot(sock, ROBOT_IP, ROBOT_PORT, cur_action)
                        start_time = time.time()
                        print("촬영 시작...")
                        swing_analysis = False
                    elapsed_time = time.time() - start_time
                    if elapsed_time < 6:   # 약 6초동안 촬영 
                        frames.append(frame_hpe)
                        current_time = datetime.now().strftime("%Y%m%d%H%M")  # 년월일시분
                        save_folder = f'{output_dir}/{current_time}/'
                        output_dir_2D = save_folder +'pose2D/'
                        os.makedirs(output_dir_2D, exist_ok=True)
                        cv2.imwrite(f"{output_dir_2D}/{frame_counter:04d}_2D.png", frame_hpe)
                        cv2.imwrite(f"{frames_folder}/{frame_counter:04d}.png", raw_frame)
                        swing_analysis_kpts_result.append(h36m_kpts.reshape(1,17, 2))
                        swing_analysis_scores_result.append(h36m_scores.reshape(1,17,1))
                        frame_counter += 1
                    else:
                        swing_analysis = True
                        cur_action = 'nothing'
                        start_time = None
                        print("촬영 종료!")
                        threading.Thread(target=event_detection_hpe).start()

                if not swing_analysis:
                    position = (120, 300)
                    x, y = position
                    cv2.rectangle(frame, (x, y - 50), (x + 300, y + 10), (0, 255, 0), -1)
                    cv2.putText(frame, "Recoding", position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

                
                cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(frame, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        
                cv2.imshow("Action", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    running = False

# 스레드 실행
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()
capture_thread.join()
process_thread.join()

cv2.destroyAllWindows()
print("프로그램 종료.")

