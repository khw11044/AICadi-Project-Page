import numpy as np
from torch.utils.data import Dataset
from common.camera import *

class SampleVideo(Dataset):
    def __init__(self, poses, width, height, mode, transform=None):
        self.poses = poses
        self.width = width
        self.height = height
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # preprocess and return frames
        poses2D = self.poses
        if self.mode==1 or self.mode==5:
            normalize_coor_pose = normalize_screen_coordinates(poses2D[:,:,:2], self.width, self.height).reshape(-1,2,17).reshape(-1,34)
        elif self.mode==2 or self.mode==6:
            normalize_coor_pose = normalize_screen_coordinates(poses2D[:,:,:2], self.width, self.height).reshape(-1,2,17)
            normalize_coor_pose = np.concatenate((normalize_coor_pose, poses2D[:,:,-1:].reshape(-1,1,17)),axis=1).reshape(-1,51)
        elif self.mode==3 or self.mode==7:
            normalize_coor_pose = canonical_normalize_coordinates(poses2D[:,:,:2])
        elif self.mode==4 or self.mode==8:
            normalize_coor_pose = canonical_normalize_coordinates(poses2D[:,:,:2])
            normalize_coor_pose = np.concatenate((normalize_coor_pose, poses2D[:,:,-1:].reshape(-1,17)),axis=1).reshape(-1,51)
        labels = np.zeros(len(poses2D)) # only for compatibility with transforms
        sample = {'inputs': normalize_coor_pose, 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample
