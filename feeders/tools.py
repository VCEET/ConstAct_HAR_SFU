import random
import matplotlib.pyplot as plt
import numpy as np
import pdb

import torch
import torch.nn.functional as F2
from PIL import Image, ImageOps
import os
import cv2
import time
from feeders.transforms_ss import *

Image.LOAD_TRUNCATED_IMAGES = True

def process_skeleton(pose_coords, pose_scores, img2aug_trans, do_flip, flip_pairs, joint_num, resized_shape):

    frame_num, person_num = pose_coords.shape[:2]
    pose_coords = pose_coords.reshape(-1,2)
    pose_2d_side = pose_2d_side.reshape(-1,2)
    pose_scores = pose_scores.reshape(-1)
    pose_3ds = pose_3ds.reshape(-1,3)
    
    # apply affine flip and affine transformation
    if do_flip:
        pose_coords[:,0] = resized_shape[1] - pose_coords[:,0] - 1
        pose_2d_side[:,0] = resized_shape[1] - pose_2d_side[:,0] - 1
        for pair in flip_pairs:
            pose_coords[pair[0], :], pose_coords[pair[1], :] = pose_coords[pair[1], :], pose_coords[pair[0], :].copy()
            pose_2d_side[pair[0], :], pose_2d_side[pair[1], :] = pose_2d_side[pair[1], :], pose_2d_side[pair[0], :].copy()
            pose_scores[pair[0]], pose_scores[pair[1]] = pose_scores[pair[1]], pose_scores[pair[0]].copy()
            pose_3ds[pair[0], :], pose_3ds[pair[1], :] = pose_3ds[pair[1], :], pose_3ds[pair[0], :].copy()

    pose_coords_xy1 = np.concatenate((pose_coords, np.ones_like(pose_coords[:,0:1])),1)
    pose_coords = np.dot(img2aug_trans, pose_coords_xy1.transpose(1,0)).transpose(1,0)[:,:2]

    pose_2d_side_xy1 = np.concatenate((pose_2d_side, np.ones_like(pose_2d_side[:,0:1])),1)
    pose_2d_side = np.dot(img2aug_trans, pose_2d_side_xy1.transpose(1,0)).transpose(1,0)[:,:2]

    # creat 3D augmenting transformation matrix
    img2aug_trans_3d = np.zeros((4,4), dtype=np.float32)
    img2aug_trans_3d[0,0:2], img2aug_trans_3d[1,0:2] = img2aug_trans[0,0:2], img2aug_trans[1,0:2]
    img2aug_trans_3d[2,2],img2aug_trans_3d[3,3] = 1,1
#    img2aug_trans_3d[0,3],img2aug_trans_3d[1,3] = img2aug_trans[0,2], img2aug_trans[1,2]

    pose_3ds_xy1 = np.concatenate((pose_3ds, np.ones_like(pose_3ds[:,0:1])),1)
    pose_3ds = np.dot(img2aug_trans_3d, pose_3ds_xy1.transpose(1,0)).transpose(1,0)[:,:3]

    # transform to input heatmap space
    pose_2d_side[:,0] = pose_2d_side[:,0] / cfg.input_img_shape[1] * cfg.input_hm_shape[1]
    pose_2d_side[:,1] = pose_2d_side[:,1] / cfg.input_img_shape[0] * cfg.input_hm_shape[0]

    pose_coords[:,0] = pose_coords[:,0] / cfg.input_img_shape[1] * cfg.input_hm_shape[1]
    pose_coords[:,1] = pose_coords[:,1] / cfg.input_img_shape[0] * cfg.input_hm_shape[0]

    pose_coords = pose_coords.reshape(frame_num, person_num, joint_num, 2)
    pose_2d_side = pose_2d_side.reshape(frame_num, person_num, joint_num, 2)
    pose_scores = pose_scores.reshape(frame_num, person_num, joint_num)
    pose_3ds = pose_3ds.reshape(frame_num, person_num, joint_num, 3)
    
    return pose_coords, pose_scores, pose_3ds, pose_2d_side

def random_scale(image):
    scale_factor = np.random.uniform(1, 1.4)  # Random scale factor between 0.7 and 1.3
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    image = cv2.resize(image, (new_w, new_h))
    image2 = image[(new_h-h)//2:h+(new_h-h)//2, (new_w-w)//2:w+(new_w-w)//2]

    return image2

def random_grayscale(image):
    if np.random.rand() < 0.2:
        image3 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image3, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels
    return image2

def load_img(path, order='RGB'):

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def valid_crop_resize(RGB_path,data_numpy,data_coords,valid_frame_num,p_interval,window,transform,img_frame_interval):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
 ##   C2, T2, V2, M2 = data_coords.shape
#    print(RGB_path,valid_frame_num)
    begin = 0
    end = valid_frame_num
    valid_size = end - begin
    image_tmpl = '{:06d}.jpg'
    #crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),window), valid_size)# constraint cropped_length lower bound as 64
        bias = np.random.randint(0,valid_size-cropped_length+1)
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]
     ##   data_2d = data_coords[:, begin+bias:begin+bias+cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)
    # resize
    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data_idx = torch.arange(begin+bias,begin+bias+cropped_length).unsqueeze(0)
    data_idx = data_idx[None,None,:,:].type(torch.float32)
    data = F2.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    data_idx = torch.floor(F2.interpolate(data_idx, size=(1,window//img_frame_interval), mode='bilinear',align_corners=False).squeeze()) # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

#    data_2d = torch.tensor(data_2d,dtype=torch.float)
#    data_2d = data_2d.permute(0, 2, 3, 1).contiguous().view(C2 * V2 * M2, cropped_length)
#    data_2d = data_2d[None, None, :, :]
#    data_2d = F2.interpolate(data_2d, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
#    data_2d = data_2d.contiguous().view(C2, V2, M2, window).permute(0, 3, 1, 2).contiguous().numpy()
    
    RGB_files = []
    for index in data_idx:
        index += 1
        path = os.path.join(RGB_path,image_tmpl.format(int(index.cpu().detach().numpy())))
        #path = os.path.join(RGB_path,str(int(index.cpu().detach().numpy()))+'.jpeg')
        RGB_image = Image.open(path)
        RGB_image = RGB_image.convert('RGB')
        RGB_files.append(RGB_image)
     
    len_RGB_files = len(RGB_files)
    RGB_files = transform(RGB_files)
    l,w,h = RGB_files.shape
    RGB_files = RGB_files.reshape(len_RGB_files,l//len_RGB_files,w,h)

   # RGB_files = torch.stack(RGB_files, dim=0)
   # RGB_files = np.stack(RGB_files, axis=0)
    
    return data, RGB_files #,data_2d

#         RGB_image = cv2.imread(os.path.join(RGB_path,image_tmpl.format(int(index.cpu().detach().numpy()))))
#         RGB_image = RGB_image.astype(np.float32) / 255.0
#         mean = [0.48145466, 0.4578275, 0.40821073]
#         std = [0.26862954, 0.26130258, 0.27577711]
#         RGB_image -= mean
#         RGB_image /= std
#         augmentations = [random_scale]

#         # Apply augmentations with a probability of 0.5 for each augmentation
#         for aug_func in augmentations:
#         #    if np.random.rand() < 0.5:
#             RGB_image = aug_func(RGB_image)

def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]

def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


def random_rot(data_numpy, theta=0.3):
    """
    data_numpy: C,T,V,M
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)  # T,3,V*M
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # T,3,3
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_torch

def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy

def load_image(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def augment_image(image, target_size=(224, 224), scale_factor=1.2, crop_size=(224, 224)):
    # Resize
    image = cv2.resize(image, target_size)
    # Scale
    # if np.random.rand() < 0.5:
    #     scale = 1, .875, .75, .66
    image = cv2.resize(image, (int(target_size[0]/scale_factor), int(target_size[1]*scale_factor)))
    
    if np.random.rand() < 0.2:
        # Random crop
        top = (image.shape[0] - crop_size[0]) // 2
        left = (image.shape[1] - crop_size[1]) // 2
        bottom = top + crop_size[0]
        right = left + crop_size[1]
        image = image[top:bottom, left:right]
    return image

def normalize_image(image):
    # Convert image to float
    image = image.astype('float32')
    # Normalize
    mean = [0.485, 0.456, 0.406]  # Mean of ImageNet dataset
    std = [0.229, 0.224, 0.225]   # Std of ImageNet dataset
    image /= 255.0
    image -= mean
    image /= std
    # Transpose image to match PyTorch tensor shape (C, H, W)
    print(image.shape)
    image = image.transpose((2, 0, 1))
    print(image.shape)
    # Convert image to PyTorch tensor
    image = torch.tensor(image)
    print(image.shape)
    print()
    return image
