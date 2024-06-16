# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

from feeders.transforms_ss import *
from randaugment import RandAugment

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

def get_augmentation(training):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = 224  #* 256 // 240
    if training:
        unique = torchvision.transforms.Compose([GroupScale((224,224)),
                                                GroupMultiScaleCrop((224,224), [1, .875, .93, .79]),
#                                                 GroupRandomHorizontalFlip(is_sth='some' in 'NTURGBD'),
#                                                 GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
#                                                                        saturation=0.2, hue=0.1),
                                                 GroupRandomGrayscale(p=0.2),
#                                                 GroupGaussianBlur(p=0.0),
#                                                 GroupSolarization(p=0.0)
                                                 ])
    else:
        unique = torchvision.transforms.Compose([GroupScale((224,224)),
                                                 GroupCenterCrop(224)
                                                 ])

    common = torchvision.transforms.Compose([Stack(roll=False),
                                             ToTorchFormatTensor(div=True),
                                             GroupNormalize(input_mean,
                                                            input_std)
                                           ])
   # resizing = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224))])

    return torchvision.transforms.Compose([ unique, common])

def randAugment(transform_train):
    print('Using RandAugment!')
    transform_train.transforms.insert(0, GroupTransform(RandAugment(2,9)))
    return transform_train
