import numpy as np

from torch.utils.data import Dataset

from feeders import tools
from feeders.Augmentation import *
import time

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
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train'] #[:100]
            self.label = np.where(npz_data['y_train']  > 0)[1] #[:100]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
            self.RGB_folders = npz_data['train_RGB'] #[:100]
        elif self.split == 'test':
            self.data = npz_data['x_test'] #[:100]
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
            self.RGB_folders = npz_data['test_RGB'] #[:100]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        
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
        RGB_folder = self.RGB_folders[index]
        RGB_path = self.img_path + RGB_folder[:]
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
