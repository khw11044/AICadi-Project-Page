import os.path as osp
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
from utils.vis import *
import copy

class GolfDB(Dataset):
    def __init__(self, data_file, vid_dir, pose_dir, seq_length, transform=None, train=True):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.pose_dir = pose_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]  # annotation info
        events = a['events']
        events -= events[0]  # now frame #s correspond to frames in preprocessed video clips
        images, poses, labels = [], [], []
        cap = cv2.VideoCapture(osp.join(self.vid_dir, '{}.mp4'.format(a['id'])))      
        poses3D_info = np.load(osp.join(self.pose_dir, 'pose3d_{}.npz'.format(a['id'])))        # poses2D_info['reconstruction'].shape
        poses3D = poses3D_info['reconstruction']
        
        if self.train:
            # random starting position, sample 'seq_length' frames
            start_frame = np.random.randint(events[-1] + 1)     # 133
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)       # 현재 FRAME, 랜덤 FRAME
            pos = start_frame
            while len(poses) < self.seq_length:
                if pos < len(poses3D):
                    poses.append(poses3D[pos])
                    # poses.append(np.array(poses2D_info[pos]['keypoints']))
                    if pos in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos)[0][0])
                    else:
                        labels.append(8)
                    pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)         # 0번째 frame부터 다시
                    pos = 0
            cap.release()
        else:
            # full clip
            # for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            for pos in range(len(poses3D)):  
                poses.append(poses3D[pos])      
                if pos in events[1:-1]:
                    labels.append(np.where(events[1:-1] == pos)[0][0])
                else:
                    labels.append(8)
            cap.release()

        sample = {'inputs':np.asarray(poses), 'labels':np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        poses, labels = sample['inputs'], sample['labels']
        # images = images.transpose((0, 3, 1, 2))
        return {'inputs': torch.from_numpy(poses).float(),
                'labels': torch.from_numpy(labels).long()}

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        poses, labels = sample['inputs'], sample['labels']
        # images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'inputs': poses, 'labels': labels}


if __name__ == '__main__':
    IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    norm = Normalize(IMAGENET_MEAN, IMAGENET_STD)  # ImageNet mean and std (RGB)
    split = 4
    dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     pose_dir='data/poses3D_160/',
                     seq_length=64, # 64
                     transform=transforms.Compose([ToTensor(), norm]),
                     train=False)   # True False

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        
    for i, sample in enumerate(data_loader):
        poses, labels = sample['inputs'], sample['labels']
        events = np.where(labels.squeeze() < 8)[0]
        print('{} events: {}'.format(len(events), events))
        
       
        poses = poses[0].detach().cpu().numpy()
        # 이미지랑 2D pose 같이 찍어보기 
        