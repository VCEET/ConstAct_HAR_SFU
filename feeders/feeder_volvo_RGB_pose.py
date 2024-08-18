import numpy as np

from torch.utils.data import Dataset

from feeders import tools_ConstAct as tools
from feeders.Augmentation import *
import time
import csv
import pandas as pd


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, img_src_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, img_frame_interval= 1):
        """
        :param data_path:
        :param label_path:
        :patam image_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """
        self.img_path = img_src_path
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.img_frame_interval = img_frame_interval
        self.header = ['j1','j2','j3','j4','j5','j6','j7','j8','j9','j10','j11','j12','j13','j14','j15','j16','j17','j18','j19','j20','j21','j22','j23','j24','j25','j26','j27','j28','j29','j30','j31','j32','j33','j34']
        self.motions = ['1_Emergency',
                        '2_Stop',
                        '3_Slow_Down',
                        '4_Come',
                        '5_BackUp',
                        '6_Left',
                        '7_Right',
                        '8_SeeMe',
                        '9_AllOK',
                        '10_FollowMe_Clapping',
                        '11_ArmCircle',
                        '12_Exc_Swing_Left',
                        '13_Exc_SwingL_Right',
                        '14_WL_Lift_Up',
                        '15_WL_Lift_Down',
                        '16_WL_Tilt_UP',
                        '17_WL_Tilt_Down']
        
        self.load_data()
        if normalization:
            self.get_mean_map()
        
    def read_dataset(self,file):    
        dtype_data = {'col1': np.float64,'col2': np.float64,'col3': np.float64,'col4': np.float64,'col5': np.float64,'col6': np.float64,'col7': np.float64,'col8': np.float64,'col9': np.float64,'col10': np.float64,'col11': np.float64,'col12': np.float64,'col13': np.float64,'col14': np.float64,'col15': np.float64,'col16': np.float64,'col17': np.float64,'col18': np.float64,'col19': np.float64,'col20': np.float64,'col21': np.float64,'col22': np.float64,'col23': np.float64,'col24': np.float64,'col25': np.float64,'col26': np.float64,'col27': np.float64,'col28': np.float64,'col29': np.float64,'col30': np.float64,'col31': np.float64,'col32': np.float64,'col33': np.float64,'col34': np.float64}
        df = pd.read_csv(file, delimiter=",", dtype=dtype_data,header=None)
        return df

    def load_data(self):
        # data: N C V T M
        div = 5
        data_motion_train = np.zeros((17,15,div,1100//div,34,3))
        data_motion_test = np.zeros((17,5,div,1100//div,34,3))
        data_motion_train_m = np.zeros((17,15,div,1100//div,25,3))
        data_motion_test_m = np.zeros((17,5,div,1100//div,25,3))
        
        label_train = np.zeros((17,15,div))
        label_test = np.zeros((17,5,div))

        mn = -1
        videos_train = []
        videos_test = []

        for m in self.motions:
            mn += 1
            train = 0
            test = 0
            for i in range(1,21):
                self.motion_path = self.data_path+m+'/'+str(i)+'/'
                self.images_path = self.motion_path + 'video/'
                self.pose_3d_path = self.motion_path + 'pose_3d_s.csv'
                data = self.read_dataset(self.pose_3d_path)

                for j in range(len(data)):
                    c_div = j // (len(data)//div)
                    c_j = j % (len(data)//div)
                    if c_div == div:
                        break
                    for k in range(len(data.columns)):
                        data_index = list(map(float, data[k][j][1:-1].split(',')))
                        if i in [1,5,9,13,17]:
                            data_motion_test[mn,test,c_div,c_j,k] = data_index
                        else:
                            data_motion_train[mn,train,c_div,c_j,k] = data_index

                    for k in range(len(data.columns)):
                        if i in [1,5,9,13,17]:
                            hip_pose = data_motion_test[mn,test,c_div,c_j,1]
                            data_motion_test[mn,test,c_div,c_j,k] = data_motion_test[mn,test,c_div,c_j,k] - hip_pose
                        else:
                            hip_pose = data_motion_train[mn,train,c_div,c_j,1]                    
                            data_motion_train[mn,train,c_div,c_j,k] = data_motion_train[mn,train,c_div,c_j,k] - hip_pose

                if i in [1,5,9,13,17]:
                    for d in range(div):
                        videos_test.append(self.images_path)
                        label_test[mn,test,d] = mn
                    test += 1
                else:
                    for d in range(div):
                        videos_train.append(self.images_path)
                        label_train[mn,train,d] = mn
                    train += 1

        ##
        data_motion_train_m[:,:,:,:,0] = data_motion_train[:,:,:,:,0]
        data_motion_train_m[:,:,:,:,1] = data_motion_train[:,:,:,:,1]
        data_motion_train_m[:,:,:,:,20] = data_motion_train[:,:,:,:,2]
        data_motion_train_m[:,:,:,:,2] = data_motion_train[:,:,:,:,26]
        data_motion_train_m[:,:,:,:,3] = data_motion_train[:,:,:,:,27]
        data_motion_train_m[:,:,:,:,4] = data_motion_train[:,:,:,:,5]
        data_motion_train_m[:,:,:,:,5] = data_motion_train[:,:,:,:,6]
        data_motion_train_m[:,:,:,:,6] = data_motion_train[:,:,:,:,7]
        data_motion_train_m[:,:,:,:,7] = data_motion_train[:,:,:,:,8]
        data_motion_train_m[:,:,:,:,21] = data_motion_train[:,:,:,:,9]
        data_motion_train_m[:,:,:,:,22] = data_motion_train[:,:,:,:,10]
        data_motion_train_m[:,:,:,:,8] = data_motion_train[:,:,:,:,12]
        data_motion_train_m[:,:,:,:,9] = data_motion_train[:,:,:,:,13]
        data_motion_train_m[:,:,:,:,10] = data_motion_train[:,:,:,:,14]
        data_motion_train_m[:,:,:,:,11] = data_motion_train[:,:,:,:,15]
        data_motion_train_m[:,:,:,:,23] = data_motion_train[:,:,:,:,16]
        data_motion_train_m[:,:,:,:,24] = data_motion_train[:,:,:,:,17]
        data_motion_train_m[:,:,:,:,12] = data_motion_train[:,:,:,:,18]
        data_motion_train_m[:,:,:,:,13] = data_motion_train[:,:,:,:,19]
        data_motion_train_m[:,:,:,:,14] = data_motion_train[:,:,:,:,20]
        data_motion_train_m[:,:,:,:,15] = data_motion_train[:,:,:,:,21]
        data_motion_train_m[:,:,:,:,16] = data_motion_train[:,:,:,:,22]
        data_motion_train_m[:,:,:,:,17] = data_motion_train[:,:,:,:,23]
        data_motion_train_m[:,:,:,:,18] = data_motion_train[:,:,:,:,24]
        data_motion_train_m[:,:,:,:,19] = data_motion_train[:,:,:,:,25]
        ##
        data_motion_test_m[:,:,:,:,0] = data_motion_test[:,:,:,:,0]
        data_motion_test_m[:,:,:,:,1] = data_motion_test[:,:,:,:,1]
        data_motion_test_m[:,:,:,:,20] = data_motion_test[:,:,:,:,2]
        data_motion_test_m[:,:,:,:,2] = data_motion_test[:,:,:,:,26]
        data_motion_test_m[:,:,:,:,3] = data_motion_test[:,:,:,:,27]
        data_motion_test_m[:,:,:,:,4] = data_motion_test[:,:,:,:,5]
        data_motion_test_m[:,:,:,:,5] = data_motion_test[:,:,:,:,6]
        data_motion_test_m[:,:,:,:,6] = data_motion_test[:,:,:,:,7]
        data_motion_test_m[:,:,:,:,7] = data_motion_test[:,:,:,:,8]
        data_motion_test_m[:,:,:,:,21] = data_motion_test[:,:,:,:,9]
        data_motion_test_m[:,:,:,:,22] = data_motion_test[:,:,:,:,10]
        data_motion_test_m[:,:,:,:,8] = data_motion_test[:,:,:,:,12]
        data_motion_test_m[:,:,:,:,9] = data_motion_test[:,:,:,:,13]
        data_motion_test_m[:,:,:,:,10] = data_motion_test[:,:,:,:,14]
        data_motion_test_m[:,:,:,:,11] = data_motion_test[:,:,:,:,15]
        data_motion_test_m[:,:,:,:,23] = data_motion_test[:,:,:,:,16]
        data_motion_test_m[:,:,:,:,24] = data_motion_test[:,:,:,:,17]
        data_motion_test_m[:,:,:,:,12] = data_motion_test[:,:,:,:,18]
        data_motion_test_m[:,:,:,:,13] = data_motion_test[:,:,:,:,19]
        data_motion_test_m[:,:,:,:,14] = data_motion_test[:,:,:,:,20]
        data_motion_test_m[:,:,:,:,15] = data_motion_test[:,:,:,:,21]
        data_motion_test_m[:,:,:,:,16] = data_motion_test[:,:,:,:,22]
        data_motion_test_m[:,:,:,:,17] = data_motion_test[:,:,:,:,23]
        data_motion_test_m[:,:,:,:,18] = data_motion_test[:,:,:,:,24]
        data_motion_test_m[:,:,:,:,19] = data_motion_test[:,:,:,:,25]
        ##
        
        if self.split == 'train':
            self.data = data_motion_train_m
            self.RGB_folders = videos_train
            self.label = label_train.reshape(-1)  
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]

        elif self.split == 'test':
            self.data = data_motion_test_m
            self.RGB_folders = videos_test
            self.label = label_test.reshape(-1)
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]

        else:
            raise NotImplementedError('data split only supports train/test')
        Nm,Nv,Ndiv,T,Nj,Ndim = self.data.shape
        self.data = self.data.reshape((Nm*Nv*Ndiv, T, 1, Nj, Ndim)).transpose(0, 4, 1, 3, 2)
        
    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        RGB_path = self.RGB_folders[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        transform_train = get_augmentation(True)
        transform_val = get_augmentation(False)

        if self.split == 'train':
            transform = transform_train
        else:
            transform = transform_val
        
        data_numpy, RGB_images = tools.valid_crop_resize(RGB_path, data_numpy, None, valid_frame_num, self.p_interval, self.window_size, transform, self.img_frame_interval)

        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        data_numpy_bone = 1
        if self.bone:
            data_numpy_bone = data_numpy
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy_bone)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy_bone[:, :, v1 - 1] - data_numpy_bone[:, :, v2 - 1]
            data_numpy_bone = bone_data_numpy

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        

        return data_numpy, label, index, RGB_images, data_numpy_bone

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)



def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
