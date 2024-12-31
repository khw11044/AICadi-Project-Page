import cv2
import requests
import numpy as np
import threading
import queue
import torch
import socket

from utils.sam2_fuc import get_predictor
from utils.yolo_fuc import get_yolo, get_bbox
from utils.tools import get_largest_bbox_from_masks
from utils.robot import control_robot

# 모델 및 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# SAM2와 YOLO 모델 초기화
sam2 = get_predictor()
yolo = get_yolo()

# 스트리밍 URL 및 상태 변수
url = "http://192.168.0.129:5000/video_front"

# 모바일 로봇의 IP 주소와 포트 설정
CENTER_IP = "192.168.0.126"  # 중앙통제시스템 서버 IP
CENTER_PORT = 5000           # 중앙통제시스템 서버 port


# 소켓 생성
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)  # 송신 버퍼 크기 설정


frame_queue = queue.Queue(maxsize=5)  # 프레임 큐
if_init = False
largest_bbox = None
running = True
seg_show=True
frame_counter = 0  # 프레임 카운터
previous_mask = None  
previous_bbox = None 
init_person_height = 600
distance = 0  # 중심 간 거리
way = 0

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
    global running, if_init, largest_bbox, seg_show, frame_counter, previous_mask, previous_bbox
    print("프레임 처리 시작...")
    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()
            height, width = frame.shape[:2]
            # print(height, width )
            # 중심점 계산
            center_x, center_y = width // 2, height // 2

            # YOLO를 통해 가장 큰 객체 감지
            if not largest_bbox:
                largest_bbox = get_bbox(frame)

                if largest_bbox:
                    x1, y1, x2, y2, _, _ = largest_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # SAM2 모델로 객체 세그멘테이션
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if largest_bbox and not if_init:
                    sam2.load_first_frame(frame)
                    bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    init_person_height = abs(y2-y1)
                    _, out_obj_ids, out_mask_logits = sam2.add_new_prompt(frame_idx=0, obj_id=1, bbox=bbox)
                    if_init = True
                    

                elif if_init:
                    if frame_counter % 5 == 0:
                        frame_counter=0
                        out_obj_ids, out_mask_logits = sam2.track(frame)
                        all_mask = torch.zeros((height, width), dtype=torch.uint8, device=device)
                        current_bbox = []  # 현재 프레임의 바운딩 박스

                        current_bbox, all_mask = get_largest_bbox_from_masks(out_obj_ids, out_mask_logits, all_mask)
                    
                        if current_bbox:
                            previous_mask = all_mask  # 현재 마스크 저장
                            previous_bbox = current_bbox  # 현재 바운딩 박스 저장
                    
                    else:
                        all_mask = previous_mask  # 이전 마스크 사용
                        current_bbox = previous_bbox  # 이전 바운딩 박스 사용
                    
                    
                    if current_bbox:
                        x1, y1, x2, y2 = current_bbox
                        
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # 바운딩 박스 중심 계산
                    box_center_x = (x1 + x2) // 2
                    box_center_y = (y1 + y2) // 2
                    person_height = abs(y2-y1)
                    person_width = abs(x2-x1)
                    distances = (init_person_height, person_height, height)
                    cv2.line(frame, (center_x, center_y), (box_center_x, box_center_y), (255, 0, 0), 2)
                    way = center_x - box_center_x
                    
                    # print('distance:', distance)
                    # print('way:',way)
                    
                    control_robot(way, distances, sock, CENTER_IP, CENTER_PORT)
                    
                    if all_mask is not None and seg_show:
                        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
                    
                    # if person_height < 50 or person_width < 50:
                    #     current_bbox = []
                    #     if_init = False
                    #     largest_bbox = None
                    #     previous_mask = None  
                    #     previous_bbox = None 
                    #     print('초기화')

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
