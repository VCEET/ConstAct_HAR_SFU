#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from model.baseline import TextCLIP, ImageCLIP
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction
from tools import *
from Text_Prompt import *
from KLLoss import KLLoss
from feeders.Text_Visual_Prompt import *
from model.Visual_Prompt import visual_prompt
from model.Fuser import Fuser

import wandb
import torch.multiprocessing

from clip.prompt_learning import PromptLearner, TextEncoder
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')

##
classes, num_text_aug, text_dict = text_prompt_openai_pasta_pool_4part()
text_list = text_prompt_openai_random()
classes_visual, num_text_aug_visual, text_dict_visual = text_visual_prompt()
text_aug_visual = text_visual_prompt_descriptive()


class_names = []

with open('text/ntu120_label_map.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        class_names.append(line.rstrip().lstrip())

#class_names = class_names[:60]

device = "cuda" if torch.cuda.is_available() else "cpu"

scaler = torch.cuda.amp.GradScaler()

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd120-cross-subject/lst_joint.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=0,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=14,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--prompt-length',
        type=int,
        default=10,
        help='the length of learnable prompt')
    parser.add_argument(
        '--class-position',
        default="middle",
        help='the position of class label in learnable prompt')
    parser.add_argument('--optimizer', default='AdamW', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--loss-alpha', type=float, default=0.8)
    parser.add_argument('--te-lr-ratio', type=float, default=0.1)
    parser.add_argument(
        '--wandb',
        type=str2bool,
        default=False,
        help='if ture logs will be saved to wandb')
    parser.add_argument('--wandb_name',default='LLM')
    parser.add_argument('--model_name',default='GAP')

    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument(
        '--load_pretrained',
        type=str2bool,
        default=False,
        help='if ture, the pretrained model will be loaded')
    

    return parser

class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.skeleton_encoder = self.skeleton_encoder.cuda(self.output_device)
        self.logit_scale = nn.Parameter(torch.ones(1,16) * np.log(1 / 0.07))


        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.skeleton_encoder = nn.DataParallel(
                    self.skeleton_encoder,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                for name in self.arg.model_args['head']:
                    self.model_text_dict[name] = nn.DataParallel(
                        self.model_text_dict[name],
                        device_ids=self.arg.device,
                        output_device=self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                for name in self.arg.model_args['head']:
                    self.model_visual_dict[name] = nn.DataParallel(
                        self.model_visual_dict[name],
                        device_ids=self.arg.device,
                        output_device=self.output_device)
                    
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        print("Number of worker is : ", self.arg.num_worker)
        print("----------------------------")
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.skeleton_encoder = Model(**self.arg.model_args)
        self.loss_ce = nn.CrossEntropyLoss().cuda(output_device)
        self.loss = KLLoss().cuda(output_device)

        if self.arg.load_pretrained:
            pretrained_path = '/localhome/mmahdavi/Mohammad_ws/human_activity_recognition/LLM_HARfusion/output/ntu60/xsub/lst_joint/Main_cocoop/'

            state_dict_sleleton_encoder = torch.load(pretrained_path+'skeleton_encoder.pt')
            state_dict_model_visual = torch.load(pretrained_path+'model_visual.pt')
            state_dict_model_text_2 = torch.load(pretrained_path+'model_text-skeleton.pt')
            state_dict_visual_encoder = torch.load(pretrained_path+'visual_encoder.pt')
            state_dict_fusion = torch.load(pretrained_path+'fusion.pt')
            state_dict_prompts = torch.load(pretrained_path+'prompts.pt')
            state_dict_meta_net = torch.load(pretrained_path+'meta_net_skeleton.pt')


        self.model_text_dict = nn.ModuleDict()
        self.model_visual_dict = nn.ModuleDict()
        self.model_text_dict_2 = nn.ModuleDict()

        for name in self.arg.model_args['head']:
            self.model_, model_state_dict = clip.load(name, device)
            self.model_text = TextEncoder(self.model_)
            self.model_text = self.model_text.cuda(self.output_device)
            self.model_text_dict[name] = self.model_text

            self.model_2, model_state_dict2 = clip.load(name, device)
            self.model_text_2 = TextEncoder(self.model_2)
            self.model_text_2 = self.model_text_2.cuda(self.output_device)
            self.model_text_dict_2[name] = self.model_text_2

            self.model_visual = ImageCLIP(self.model_)

            self.model_visual = self.model_visual.cuda(self.output_device)
            self.model_visual_dict[name] = self.model_visual

            self.visual_encoder = visual_prompt("Transf",model_state_dict,8).cuda(self.output_device)
            self.fusion = Fuser(self.output_device,self.arg.model_args['num_class']).cuda(self.output_device)

            self.prompt_learner2 = PromptLearner(class_names, self.model_2,self.arg.prompt_length,self.arg.class_position).cuda(self.output_device)
            self.tokenized_prompts2 = self.prompt_learner2.tokenized_prompts.cuda(self.output_device)
            self.prompts2 = self.prompt_learner2().unsqueeze(0).repeat(num_text_aug,1,1,1)

            self.tokenized_prompts2 = self.tokenized_prompts2.unsqueeze(0).repeat(num_text_aug,1,1)
            
            ## Added for conditional prompt learning
            self.meta_net_visual = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(512, 512 // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(512 // 16, self.arg.prompt_length))
            ])).cuda(self.output_device)
            self.meta_net_skeleton = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(512, 512 // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(512 // 16, self.arg.prompt_length))
            ])).cuda(self.output_device)
            ##

            if self.arg.load_pretrained:
                self.skeleton_encoder.load_state_dict(state_dict_sleleton_encoder)
                self.model_visual.load_state_dict(state_dict_model_visual)
                self.model_text_2.load_state_dict(state_dict_model_text_2)
                self.visual_encoder.load_state_dict(state_dict_visual_encoder)
                self.fusion.load_state_dict(state_dict_fusion)
                self.prompt_learner2.load_state_dict(state_dict_prompts)
                self.meta_net_skeleton.load_state_dict(state_dict_meta_net)

                self.model_text_dict_2[name] = self.model_text_2.cuda(self.output_device)


        if self.arg.weights:
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.skeleton_encoder.load_state_dict(weights)
            except:
                state = self.skeleton_encoder.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.skeleton_encoder.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'AdamW':
            vision_params = list(map(id, self.model_.visual.parameters()))
            text_params = filter(lambda p: id(p) not in vision_params,
                                 self.model_.parameters())

            self.optimizer = optim.AdamW([{'params': self.skeleton_encoder.parameters(),'lr': self.arg.base_lr*10},
                 {'params': self.visual_encoder.parameters(),'lr': self.arg.base_lr*10},
                 {'params': self.model_.visual.parameters(), 'lr': self.arg.base_lr},
                 {'params': self.fusion.parameters(), 'lr':self.arg.base_lr*10},
                 {'params': self.model_text_2.parameters(), 'lr':self.arg.base_lr},
                 {'params': self.prompt_learner2.parameters(), 'lr':self.arg.base_lr*10},
                 {'params': self.meta_net_skeleton.parameters(), 'lr':self.arg.base_lr*10},],
                betas=(0.9, 0.98),lr=self.arg.base_lr, eps=1e-8,
                weight_decay=self.arg.weight_decay)
                                 
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'AdamW':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.skeleton_encoder.train()
        self.model_visual.train()
        self.visual_encoder.train()
        self.fusion.train()
        self.model_text_2.train()
        self.prompt_learner2.train()
        self.meta_net_skeleton.train()
        
        ## You may add model_text or model_visual to the optimizer

        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_total_value = []
        loss_cls_value = []
        loss_pose_value = []
        loss_visual_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        t_start = time.time()

        for batch_idx, (data, label, index, RGB_images, data_bone) in enumerate(process):       

            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
            timer['dataloader'] += self.split_time()
            self.optimizer.zero_grad()
            
            label_g = gen_label(label)
            label = label.long().cuda(self.output_device)

            # forward
            with torch.cuda.amp.autocast():

                ## Visual Encoding
                b,t,c,h,w = RGB_images.size()
                RGB_images= RGB_images.float().to(device)
                image_embedding = self.model_visual(RGB_images.reshape(-1,c,h,w)) 

                image_embedding = image_embedding.view(b,t,-1)
                visual_features, image_embedding = self.visual_encoder(image_embedding)

                ## Text generation
                ## Skeleton Encoding
                skeleton_features, feature_dict, logit_scale, part_feature_list = self.skeleton_encoder(data)

                ## Text generation skeleton
                text_embedding_skeleton_list = []
                for ind in range(num_text_aug):              

                    if ind > 0:
                        skeleton_prompt_feat = self.meta_net_skeleton(part_feature_list[ind-1]).unsqueeze(-1)
                        self.prompts2[ind,label][:,1:1+self.arg.prompt_length] += skeleton_prompt_feat
                    else:
                        skeleton_prompt_feat = self.meta_net_skeleton(feature_dict['ViT-B/16']).unsqueeze(-1)
                        self.prompts2[ind,label][:,1:1+self.arg.prompt_length] += skeleton_prompt_feat
                    
                    text_embedding_skeleton = self.model_text_dict_2[self.arg.model_args['head'][0]](self.prompts2[ind,label],self.tokenized_prompts2[ind,label]).float()
                    text_embedding_skeleton_list.append(text_embedding_skeleton)

                ## loss calculations
                # skeleton loss
                loss_te_list = []
                for ind in range(num_text_aug):
                    if ind == 0:
                        logits_per_image, logits_per_text = create_logits(feature_dict[self.arg.model_args['head'][0]],text_embedding_skeleton_list[ind],logit_scale[:,0].mean())
                        ground_truth = torch.tensor(label_g,dtype=feature_dict[self.arg.model_args['head'][0]].dtype,device=device)
                    else:
                        logits_per_image, logits_per_text = create_logits(part_feature_list[ind-1],text_embedding_skeleton_list[ind],logit_scale[:,ind].mean())
                        ground_truth = torch.tensor(label_g,dtype=part_feature_list[ind-1].dtype,device=device)
                    loss_imgs = self.loss(logits_per_image,ground_truth)
                    loss_texts = self.loss(logits_per_text,ground_truth)
                    loss_te_list.append((loss_imgs + loss_texts) / 2)
                loss_pose = sum(loss_te_list) / len(loss_te_list)

                ## Fusing the two modalities
                output = self.fusion(visual_features,skeleton_features)

                ## Calc loss
                loss_ce = self.loss_ce(output, label)
                loss = loss_ce + self.arg.loss_alpha*loss_pose
        
            scaler.scale(loss).backward()

            scaler.step(self.optimizer)
            scaler.update()

            loss_total_value.append(loss.data.item())
            loss_pose_value.append(loss_pose.item())
            loss_cls_value.append(loss_ce.item())

            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            
            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

            # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_total_value), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict_sleleton_encoder = self.skeleton_encoder.state_dict()
            state_dict_model_visual = self.model_visual.state_dict()
            state_dict_model_text_2 = self.model_text_2.state_dict()
            state_dict_visual_encoder = self.visual_encoder.state_dict()
            state_dict_fusion = self.fusion.state_dict()
            state_dict_prompt_learner2 = self.prompt_learner2.state_dict()
            state_dict_meta_net_skeleton = self.meta_net_skeleton.state_dict()
           
            epoch_path = self.arg.work_dir+'/'+str(epoch)
            os.makedirs(epoch_path, exist_ok=True)
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_sleleton_encoder.items()])
            torch.save(weights, epoch_path + '/skeleton_encoder' + '.pt')
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_visual_encoder.items()])
            torch.save(weights, epoch_path + '/visual_encoder' + '.pt')
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_model_visual.items()])
            torch.save(weights, epoch_path +  '/model_visual' + '.pt')
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_model_text_2.items()])
            torch.save(weights, epoch_path + '/model_text-skeleton' + '.pt')
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_fusion.items()])
            torch.save(weights, epoch_path + '/fusion' + '.pt')
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_prompt_learner2.items()])
            torch.save(weights, epoch_path + '/prompts' + '.pt')
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_meta_net_skeleton.items()])
            torch.save(weights, epoch_path + '/meta_net_skeleton' + '.pt')


        elapsed_time_train = time.time()-t_start
        self.log['train_loss_cls'].append(np.mean(loss_cls_value))
        self.log['train_loss_pose'].append(np.mean(loss_pose_value))
        self.log['train_loss_total'].append(np.mean(loss_total_value))

        self.log['lrate'].append(self.lr)
        self.log['elapsed_time_train'].append(elapsed_time_train)


    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.skeleton_encoder.eval()
        self.visual_encoder.eval()
        self.fusion.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)

            t_start = time.time()

            for batch_idx, (data, label, index, RGB_images, data_bone) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    b, _, _, _, _ = data.size()
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    RGB_images = RGB_images.float().cuda(self.output_device)

                    b,t,c,h,w = RGB_images.size()
                    image_embedding = self.model_visual(RGB_images.reshape(-1,c,h,w)) 
                    image_embedding = image_embedding.view(b,t,-1)
                    visual_features, image_embedding = self.visual_encoder(image_embedding)

                    skeleton_features, _, _, _ = self.skeleton_encoder(data)

                    output = self.fusion(visual_features,skeleton_features)

                    loss = self.loss_ce(output, label)

                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            self.log['val_loss'].append(loss)

            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            self.log['Accuracy'].append(accuracy)

            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))
                
            elapsed_time_val = time.time() - t_start
            self.log['elapsed_time_val'].append(elapsed_time_val)

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def start(self):
        self.log = OrderedDict([
                        ('epoch', []),
                        ('train_loss_total', []),
                        ('train_loss_cls', []),
                        ('train_loss_pose', []),
                        ('train_loss_visual', []),
                        ('lrate', []),
                        ('elapsed_time_train', []),
                        ('val_loss', []),
                        ('Accuracy', []),
                        ('elapsed_time_val', []),
                ])
        
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            start_epoch = 0
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)
                
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

                if self.arg.wandb:
                    dic = {x: v[-1] for x,v in self.log.items() if v }
                    wandb.log(dic)
                    
            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.visual_encoder.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.visual_encoder.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.full_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)

    if arg.wandb:
        wandb.init(project=arg.wandb_name,  entity="transfuser", name = arg.model_name) 
    processor = Processor(arg)
    processor.start()
