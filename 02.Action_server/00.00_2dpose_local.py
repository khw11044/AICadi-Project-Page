import cv2
from ultralytics import YOLO
import numpy as np
from actiontools.tools import *
import os

# YOLO Pose 모델 로드
model = YOLO("./models/yolo11n-pose.pt")  # 원하는 모델을 다운로드 후 경로 수정 가능


def extract_keypoints(results):
    results = results.reshape(17, 3)
    pose = np.array([[res[0], res[1], res[2]] for idx, res in enumerate(results)]).flatten()
    return pose


# 웹캠 캡처 초기화
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# 비디오 저장을 위한 설정
output_file = 'output_pose.mp4'  # 저장될 비디오 파일 이름
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 비디오 코덱 설정 (mp4 코덱)
fps = 20.0  # 초당 프레임 수 설정
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 프레임 너비
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 높이
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))  # VideoWriter 객체 초기화

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    frame_size = frame.shape

    # try:
    results = model(frame, verbose=False)

    # 결과에서 keypoints 가져오기
    for result in results:
        jointsdata = result.keypoints.data.cpu().numpy()
        keypoints, scores = jointsdata[:1, :, :2], jointsdata[:1, :, 2:]
        if scores.any():
            h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(keypoints, scores)
            kps = np.concatenate([h36m_kpts, h36m_scores], axis=2)
            frame = show2Dpose(h36m_kpts, frame)
            frame= drawkeypoints(jointsdata, frame)
            frame = show2Dpose_h36m(kps,frame)

    # except Exception as e:
    #     print('Error:', e)

    # 비디오 프레임 저장
    out.write(frame)

    # 프레임 출력
    cv2.imshow('Pose Estimation', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
out.release()  # 비디오 저장 객체 해제
cv2.destroyAllWindows()
