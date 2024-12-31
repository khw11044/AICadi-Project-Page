from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
import argparse
import time
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.backends.cudnn as cudnn
import cv2
import copy

from lib.hrnet.lib.utils.utilitys import plot_keypoint, PreProcess, write, load_json
from lib.hrnet.lib.config import cfg, update_config
from lib.hrnet.lib.utils.transforms import *
from lib.hrnet.lib.utils.inference import get_final_preds
from lib.hrnet.lib.models import pose_hrnet

cfg_dir = 'demo/lib/hrnet/experiments/'
model_dir = 'demo/lib/checkpoint/'

# Loading human detector model
from lib.yolov3.human_detector import load_model as yolo_model
from lib.yolov3.human_detector import yolo_human_det as yolo_det
from lib.sort.sort import Sort


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default=cfg_dir + 'w48_384x288_adam_lr1e-3.yaml',
                        help='experiment configure file name')
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None,
                        help="Modify config options using the command-line")
    parser.add_argument('--modelDir', type=str, default=model_dir + 'pose_hrnet_w48_384x288.pth',
                        help='The model directory')
    parser.add_argument('--det-dim', type=int, default=416,
                        help='The input dimension of the detected image')
    parser.add_argument('--thred-score', type=float, default=0.30,
                        help='The threshold of object Confidence')
    parser.add_argument('-a', '--animation', action='store_true',
                        help='output animation')
    parser.add_argument('-np', '--num-person', type=int, default=1,
                        help='The maximum number of estimated poses')
    parser.add_argument("-v", "--video", type=str, default='camera',
                        help="input video file name")
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args = parser.parse_args()

    return args


def reset_config(args):
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


# load model
def model_load(config):
    model = pose_hrnet.get_pose_net(config, is_train=False)
    if torch.cuda.is_available():
        model = model.cuda()

    state_dict = torch.load(config.OUTPUT_DIR)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k  # remove module.
        #  print(name,'\t')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    # print('HRNet network successfully loaded')
    
    return model


def gen_video_kpts(video, det_dim=416, num_peroson=1, gen_output=False):
    # Updating configuration
    args = parse_args()
    reset_config(args)

    cap = cv2.VideoCapture(video)

    # Loading detector and pose model, initialize sort for track
    human_model = yolo_model(inp_dim=det_dim)
    pose_model = model_load(cfg)
    people_sort = Sort(min_hits=0)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    kpts_result = []
    scores_result = []
    for ii in tqdm(range(video_length)):
        ret, frame = cap.read()

        if not ret:
            continue

        bboxs, scores = yolo_det(frame, human_model, reso=det_dim, confidence=args.thred_score)

        if bboxs is None or not bboxs.any():
            print('No person detected!')
            if ii==0:
                bboxs = np.array([[ 59.42,  45.88, 106.24, 157.58]])
                scores = np.array([[0.1992217]])
                bboxs_pre = copy.deepcopy(bboxs) 
                scores_pre = copy.deepcopy(scores) 
            else:
                bboxs = bboxs_pre
                scores = scores_pre
        else:
            bboxs_pre = copy.deepcopy(bboxs) 
            scores_pre = copy.deepcopy(scores) 

        # Using Sort to track people
        people_track = people_sort.update(bboxs)

        # Track the first two people in the video and remove the ID
        if people_track.shape[0] == 1:
            people_track_ = people_track[-1, :-1].reshape(1, 4)
        elif people_track.shape[0] >= 2:
            people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
            people_track_ = people_track_[::-1]
        else:
            continue

        track_bboxs = []
        for bbox in people_track_:
            bbox = [round(i, 2) for i in list(bbox)]
            track_bboxs.append(bbox)

        with torch.no_grad():
            # bbox is coordinate location
            inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, cfg, num_peroson)

            inputs = inputs[:, [2, 1, 0]]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
            output = pose_model(inputs)

            # compute coordinate
            preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

        kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
        scores = np.zeros((num_peroson, 17), dtype=np.float32)
        for i, kpt in enumerate(preds):
            kpts[i] = kpt

        for i, score in enumerate(maxvals):
            scores[i] = score.squeeze()

        kpts_result.append(kpts)
        scores_result.append(scores)

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)

    keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
    scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)

    return keypoints, scores

import glob 

def get_frame_images(frames_dir):
    """frames 폴더에서 이미지들을 읽고 정렬된 순서로 반환"""
    image_paths = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
    return image_paths

def gen_frames_kpts(frames_folder, det_dim=416, num_peroson=1, gen_output=False):
    # Updating configuration
    args = parse_args()  # 기존 함수와 동일한 설정 파라미터
    reset_config(args)

    # Load detector and pose model, initialize SORT for tracking
    human_model = yolo_model(inp_dim=det_dim)
    pose_model = model_load(cfg)
    people_sort = Sort(min_hits=0)

    # Load all image files from the frames folder
    frame_files = get_frame_images(frames_folder)
    
    total_frames = len(frame_files)
    kpts_result = []
    scores_result = []

    bboxs_pre = None
    scores_pre = None
    frames = []
    for ii, frame_path in tqdm(enumerate(frame_files), total=total_frames):
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Error reading frame: {frame_path}")
            continue
        frames.append(frame)
        bboxs, scores = yolo_det(frame, human_model, reso=det_dim, confidence=args.thred_score)

        if bboxs is None or not bboxs.any():
            print(f'No person detected in frame {ii}!')
            if ii == 0:
                bboxs = np.array([[59.42, 45.88, 106.24, 157.58]])
                scores = np.array([[0.1992217]])
                bboxs_pre = copy.deepcopy(bboxs)
                scores_pre = copy.deepcopy(scores)
            else:
                bboxs = bboxs_pre
                scores = scores_pre
        else:
            bboxs_pre = copy.deepcopy(bboxs)
            scores_pre = copy.deepcopy(scores)

        # Using SORT to track people
        people_track = people_sort.update(bboxs)

        # Track the first N people (default num_peroson=1)
        if people_track.shape[0] == 1:
            people_track_ = people_track[-1, :-1].reshape(1, 4)
        elif people_track.shape[0] >= 2:
            people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
            people_track_ = people_track_[::-1]
        else:
            continue

        track_bboxs = []
        for bbox in people_track_:
            bbox = [round(i, 2) for i in list(bbox)]
            track_bboxs.append(bbox)

        with torch.no_grad():
            # Preprocess input for pose estimation
            inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, cfg, num_peroson)
            inputs = inputs[:, [2, 1, 0]]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
            output = pose_model(inputs)

            # Compute keypoint coordinates and scores
            preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

        kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
        scores = np.zeros((num_peroson, 17), dtype=np.float32)
        for i, kpt in enumerate(preds):
            kpts[i] = kpt

        for i, score in enumerate(maxvals):
            scores[i] = score.squeeze()

        kpts_result.append(kpts)
        scores_result.append(scores)

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)

    keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
    scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)
    width, height, _ = frame.shape
    return keypoints, scores, width, height, frames


# YOLO Pose 모델 로드
from ultralytics import YOLO
from actiontools.tools import *

def gen_frames_yolo(frames_folder):
    # Updating configuration
    args = parse_args()  # 기존 함수와 동일한 설정 파라미터
    reset_config(args)

    pose_model = YOLO("./models/yolo11n-pose.pt")  # 원하는 모델을 다운로드 후 경로 수정 가능

    # Load all image files from the frames folder
    frame_files = get_frame_images(frames_folder)
    
    total_frames = len(frame_files)
    kpts_result = []
    scores_result = []

    bboxs_pre = None
    scores_pre = None
    frames = []
    for ii, frame_path in tqdm(enumerate(frame_files), total=total_frames):
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Error reading frame: {frame_path}")
            continue
        frame_size = frame.shape
        frames.append(frame)
        
        results = pose_model(frame, verbose=False)
        
        for result in results:
            jointsdata = result.keypoints.data.cpu().numpy() 
            keypoints, scores = jointsdata[:1,:,:2], jointsdata[:1,:,2:]
            if scores.any():
                h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(keypoints, scores)

        bboxs = results[0].boxes.data.cpu().numpy() 
        if bboxs is None or not bboxs.any():
            print(f'No person detected in frame {ii}!')
            if ii == 0:
                bboxs = np.array([[59.42, 45.88, 106.24, 157.58]])
                scores = np.array([[0.1992217]])
                bboxs_pre = copy.deepcopy(bboxs)
                scores_pre = copy.deepcopy(scores)
            else:
                bboxs = bboxs_pre
                scores = scores_pre
        else:
            bboxs_pre = copy.deepcopy(bboxs)
            scores_pre = copy.deepcopy(scores)


        # kpts = np.zeros((1, 17, 2), dtype=np.float32)
        # scores = np.zeros((1, 17), dtype=np.float32)
        
        kpts = h36m_kpts.reshape(1,17, 2)
        scores = h36m_scores.reshape(1,17)
        
        kpts_result.append(kpts)
        scores_result.append(scores)

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)

    keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
    scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)
    width, height, _ = frame.shape
    return keypoints, scores, width, height, frames