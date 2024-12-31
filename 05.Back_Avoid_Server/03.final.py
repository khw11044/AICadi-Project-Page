import socket
import cv2
import requests
import numpy as np
import threading
import queue
import torch
import matplotlib

from depth_anything_v2.dpt import DepthAnythingV2

# 모델 및 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

input_size = 518
encoder = 'vits'  # ['vits', 'vitb', 'vitl', 'vitg']
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

depth_anything = DepthAnythingV2(**model_configs[encoder])
depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()

cmap = matplotlib.colormaps.get_cmap('Spectral_r')
url = "http://192.168.0.129:5000/video_feed"  # 스트리밍 URL
frame_queue = queue.Queue(maxsize=5)
running = True

# 정지 신호를 보내는 함수 (UDP)
def send_stop_signal():
    stop_signal_address = ('192.168.0.123', 12345)  # 로봇 IP 및 포트
    stop_signal = "STOP"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(stop_signal.encode(), stop_signal_address)
    sock.close()
    print("정지 신호 전송 완료")

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

# 프레임을 처리하고 결과를 시각화
def process_frames():
    global running
    print("프레임 처리 시작...")
    reference_depth_value = 10  # 하단 1/6 지점 기준 거리 (10cm)

    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_height, frame_width = frame.shape[:2]

            # 깊이 맵 추정 및 정제
            depth = depth_anything.infer_image(frame, input_size)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            refined_depth = cv2.medianBlur(depth, 5)  # 단일 정제 단계

            # 하단 1/6 지점과 중앙 지점 설정
            target_x, target_y = frame_width // 2, frame_height * 5 // 6  # 하단 1/6 지점
            mid_x, mid_y = frame_width // 2, frame_height // 2  # 중앙 지점

            # 기준 거리로 캘리브레이션 (역수로 거리 계산)
            target_depth = refined_depth[target_y, target_x]  # 하단 1/6 지점의 깊이
            mid_depth = refined_depth[mid_y, mid_x]  # 중앙 지점의 깊이
            
            # 뎁스 값이 0인 경우 처리
            calibrated_mid_depth = (255 / mid_depth) * reference_depth_value if mid_depth != 0 else float('inf')
            calibrated_target_depth = (255 / target_depth) * reference_depth_value if target_depth != 0 else float('inf')

            # 시각화
            depth_color = (cmap(refined_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            # 하단 1/6 지점 시각화
            cv2.circle(depth_color, (target_x, target_y), 5, (255, 255, 255), -1)
            cv2.putText(depth_color, f'Depth: {target_depth}', (target_x + 10, target_y - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(depth_color, f'Calib Dist: {calibrated_target_depth:.2f}cm', (target_x + 10, target_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # 중앙 지점 시각화
            cv2.circle(depth_color, (mid_x, mid_y), 5, (0, 255, 0), -1)
            cv2.putText(depth_color, f'Depth: {mid_depth}', (mid_x + 10, mid_y - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(depth_color, f'Calib Dist: {calibrated_mid_depth:.2f}cm', (mid_x + 10, mid_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # 중앙 지점이 16cm 이내로 접근했을 때 정지 신호 보내기
            if calibrated_mid_depth <= 12.0:
                print("중앙 지점 16cm 이내로 접근: 정지 신호 전송")
                send_stop_signal()

            # 결과 표시
            combined_frame = cv2.hconcat([frame, np.ones((frame_height, 50, 3), dtype=np.uint8) * 255, depth_color])
            cv2.imshow("Original + Depth Map", combined_frame)

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