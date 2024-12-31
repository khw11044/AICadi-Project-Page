import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os 
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy
from IPython import embed

sys.path.append(os.getcwd())
from common.model_poseformer import PoseTransformerV2 as Model
from common.camera import *

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def get_pose2D(video_path, video_name, input_dir):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('Generating 2D pose...')
    keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    # 추가
    scores = scores.reshape(scores.shape[0],scores.shape[1],scores.shape[2],1)
    kps_conf = np.concatenate([keypoints, scores], axis=3)
    # re_kpts = revise_kpts(keypoints, scores, valid_frames)

    # input_dir += 'input_2D/'
    # os.makedirs(input_dir, exist_ok=True)

    output_npz = input_dir + '{}.npz'.format(video_name)
    np.savez_compressed(output_npz, reconstruction=kps_conf)



def get_pose3D(video_path, video_name, input_dir, output_dir, model_name):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.embed_dim_ratio, args.depth, args.frames = 32, 4, 243
    args.number_of_kept_frames, args.number_of_kept_coeffs = 27, 27
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/'
    args.n_joints, args.out_joints = 17, 17

    ## Reload 
    model = nn.DataParallel(Model(args=args)).cuda()

    model_dict = model.state_dict()
    # Put the pretrained model of PoseFormerV2 in 'checkpoint/']
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, model_name)))[0]

    pre_dict = torch.load(model_path)
    model.load_state_dict(pre_dict['model_pos'], strict=True)

    model.eval()

    ## input
    kps_conf = np.load(input_dir + '{}.npz'.format(video_name), allow_pickle=True)['reconstruction']
    keypoints, scores = kps_conf[:,:,:,:2], kps_conf[:,:,:,2:]
    
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ## 3D
    post_outs = []
    print('Generating 3D pose...')
    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        if img is None:
            continue
        img_size = img.shape

        ## input frames
        start = max(0, i - args.pad)
        end =  min(i + args.pad, len(keypoints[0])-1)

        input_2D_no = keypoints[0][start:end+1]
        
        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != args.frames:
            if i < args.pad:
                left_pad = args.pad - i
            if i > len(keypoints[0]) - args.pad - 1:
                right_pad = i + args.pad - (len(keypoints[0]) - 1)

            input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')
        
        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        # input_2D_no += np.random.normal(loc=0.0, scale=5, size=input_2D_no.shape)
        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[ :, :, 0] *= -1
        input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        # (2, 243, 17, 2)
        
        input_2D = input_2D[np.newaxis, :, :, :, :]

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

        N = input_2D.size(0)

        ## estimation
        output_3D_non_flip = model(input_2D[:, 0]) 
        output_3D_flip     = model(input_2D[:, 1])
        # [1, 1, 17, 3]

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()
        # post_out[:,1], post_out[:,2] = post_out[:,2], -post_out[:,1]
        
        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])
        
        post_outs.append(post_out)
        
    # output_dir += 'output_3D/'
    # os.makedirs(output_dir, exist_ok=True)
    output_3Dposes = np.array(post_outs)
    output_npz = output_dir + 'pose3d_{}.npz'.format(video_name)
    np.savez_compressed(output_npz, reconstruction=output_3Dposes)
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_path', type=str, default='./data/videos_160', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    parser.add_argument('--model_name', type=str, default='27_243_45.2.bin', help='input video')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 추가
    input_dir = './data/pose2D/'
    output_dir = './data/pose3D/'
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 추가
    all_videos = sorted(glob.glob(args.videos_path + '/*.mp4'))
    num_of_all_videos = len(all_videos)
    for i, video_path in enumerate(all_videos):
        video_name = video_path.split('/')[-1].split('.')[0]
        print(video_name, '({}/{}) video processing....'.format(i,num_of_all_videos))
        # 파라메타 바꿈
        get_pose2D(video_path, video_name, input_dir)
        get_pose3D(video_path, video_name, input_dir, output_dir, args.model_name)
        # img2video(video_path, output_dir)



