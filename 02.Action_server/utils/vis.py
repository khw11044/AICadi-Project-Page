import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os 
from tqdm import tqdm



def show2Dpose(kps, img, ax):
    j_num = kps.shape[0]
    if j_num == 17:
        connections = [ [0, 1], [1, 2], [2, 3],              # 0이 root joint
                        [0, 4], [4, 5], [5, 6],
                        [0, 7], [7, 8], [8, 9], [9, 10],
                        [8, 11], [11, 12], [12, 13],
                        [8, 14], [14, 15], [15, 16]]
    else:
        connections = [ [0, 1], [1, 2], [2, 3],              # 0이 root joint
                        [0, 4], [4, 5], [5, 6],
                        [0, 7], [7, 8], [8, 9],
                        [7, 10], [10, 11], [11, 12],
                        [7, 13], [13, 14], [14, 15]]

        
    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    ccolor = (0, 150, 0)
    thickness = 3

    for j, c in enumerate(connections):
        start = map(int, kps[c[0]])     # (17, 2)
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        if j in [0, 1, 2, 13, 14, 15]:
            color = rcolor
        elif j in [3, 4, 5, 10, 11, 12]:
            color = lcolor
        else:
            color = ccolor

        cv2.line(img, (start[0], start[1]), (end[0], end[1]), color, thickness)
        cv2.circle(img, (start[0], start[1]),thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)
    # idx = 14
    # cv2.circle(img, (int(kps[idx][0]), int(kps[idx][1])), thickness=-1, color=(255, 255, 0), radius=5) 
    
    ax.imshow(img)
    ax.axis('off')
                
    # return img

def return2Dpose(kps, img):
    j_num = kps.shape[0]
    if j_num == 17:
        connections = [ [0, 1], [1, 2], [2, 3],              # 0이 root joint
                        [0, 4], [4, 5], [5, 6],
                        [0, 7], [7, 8], [8, 9], [9, 10],
                        [8, 11], [11, 12], [12, 13],
                        [8, 14], [14, 15], [15, 16]]
    else:
        connections = [ [0, 1], [1, 2], [2, 3],              # 0이 root joint
                        [0, 4], [4, 5], [5, 6],
                        [0, 7], [7, 8], [8, 9],
                        [7, 10], [10, 11], [11, 12],
                        [7, 13], [13, 14], [14, 15]]


    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    ccolor = (0, 150, 0)
    thickness = 2
    radius=2

    for j, c in enumerate(connections):
        start = map(int, kps[c[0]]) # kps[c[0]]
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        if j in [0, 1, 2, 13, 14, 15]:
            color = rcolor
        elif j in [3, 4, 5, 10, 11, 12]:
            color = lcolor
        else:
            color = ccolor

        cv2.line(img, (start[0], start[1]), (end[0], end[1]), color, thickness)
        cv2.circle(img, (start[0], start[1]),thickness=-1, color=(0, 255, 0), radius=radius)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=radius)

    return img


def show3Dgt(vals, ax, RADIUS):

    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)
    ccolor = 'g'

    I = np.array([0, 0, 1, 4, 2, 5,  0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array([1, 4, 2, 5, 3, 6,  7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    RL = np.array([0, 1, 0, 1, 0, 1,  2, 2, 0,   1,  0,  0,  1,  1, 2, 2])

    for i in np.arange(len(I)):
        if RL[i] == 0:
            color = rcolor
        elif RL[i] == 1:
            color = lcolor
        else:
            color = ccolor
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=color)

    # RADIUS = 0.72
    # RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_aspect('equal')  # works fine in matplotlib==2.2.2

    ax.view_init(elev=15., azim=70)

    # background color
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)


def show3Dpose(vals, ax, RADIUS, angles=(20,-70)):

    rcolor = (0, 0, 1)
    lcolor = (1, 0, 0)
    ccolor = 'g'

    I = np.array([0, 0, 1, 4, 2, 5,  0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array([1, 4, 2, 5, 3, 6,  7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    RL = np.array([0, 1, 0, 1, 0, 1,  2, 2, 0,   1,  0,  0,  1,  1, 2, 2])

    for i in np.arange(len(I)):
        if RL[i] == 0:
            color = rcolor 
        elif RL[i] == 1:
            color = lcolor
        else:
            color = ccolor
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        # ax.plot(x, y, z, lw=2, color=color)
        ax.plot(x, z, -y, lw=2, color=color)


    # xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    xroot, yroot, zroot = vals[0, 0], vals[0, 2], -vals[0, 1]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    # ax.set_aspect('equal')  # works fine in matplotlib==2.2.2

    # angle
    ax.view_init(elev=angles[0], azim=angles[1])

    # background color
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)


def show3Dpose_with_gt(vals, gt, ax, RADIUS,angles=(20,-70)):

    rcolor = (0, 0, 1)
    lcolor = (1, 0, 0)
    ccolor = 'g'

    I = np.array([0, 0, 1, 4, 2, 5,  0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array([1, 4, 2, 5, 3, 6,  7, 8, 14, 11, 15, 16, 12, 13, 9, 10])


    RL = np.array([0, 1, 0, 1, 0, 1,  2, 2, 0,   1,  0,  0,  1,  1, 2, 2])

    for i in np.arange(len(I)):
        if RL[i] == 0:
            color = rcolor
        elif RL[i] == 1:
            color = lcolor
        else:
            color = ccolor
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        # ax.plot(x, y, z, lw=2, color=color)
        ax.plot(x, z, -y, lw=2, color=color)
        
        gt_x, gt_y, gt_z = [np.array([gt[I[i], j], gt[J[i], j]]) for j in range(3)]
        # ax.plot(gt_x, gt_y, gt_z, lw=2, color=(0,0,0))
        ax.plot(gt_x, gt_z, -gt_y, lw=2, color=(0,0,0))

    # RADIUS = 0.72
    # RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    # ax.set_aspect('equal')  # works fine in matplotlib==2.2.2

    # angle
    ax.view_init(elev=angles[0], azim=angles[1])

    # background color
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)
    

def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def img2video(video_path, output_dir, video_name):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size) 
    print('Generating demo video ...')
    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()


def synthesis2D_3Dimages(output_dir_2D,output_dir_3D, output_dir_pose):
    ## all
    save_dir = output_dir_pose + 'pose/' 
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\nGenerating demo synthesis ...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        # image_2d = image_2d[:, edge:image_2d.shape[1] - edge]
        
        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(16, 8))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = font_size)

        
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')
        
        
# import matplotlib.pyplot as plt 
# from mpl_toolkits.mplot3d import Axes3D

        
# def save_3dpose_img(fig, epoch, post_out, post_gt):
#     for v in range(len(post_out)):
#         ax = fig.add_subplot(1,4,1+v,projection='3d',aspect='auto')
#         post_out *= 10
#         RADIUS = 15
#         show3Dpose_with_gt(post_out[v], post_gt[v], ax, RADIUS, angles=(10,-80))

#     output_dir_3D = './pose3D/'
#     os.makedirs(output_dir_3D, exist_ok=True)
#     plt.savefig(output_dir_3D + str(('%04d'% epoch)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')
