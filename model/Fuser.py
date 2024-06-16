import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt

class Fuser(nn.Module):
    def __init__(self,output_device,num_class):
        super(Fuser,self).__init__()
        self.fc = nn.Linear(512,num_class).cuda(output_device)
        self.adjuster = nn.Linear(256,512).cuda(output_device)
        self.attn_skel = nn.MultiheadAttention(512,8,batch_first=True).cuda(output_device)
        self.attn_img = nn.MultiheadAttention(512,8,batch_first=True).cuda(output_device)
        self.bn = nn.BatchNorm1d(200).cuda(output_device)
        self.upsample = nn.Linear(16,100).cuda(output_device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4).cuda(output_device)
        self.pick_best_skel = self.refined_downsample(100,3)
        self.pick_best_img = self.refined_downsample(16,2)


    def get_conv2d(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        if type(kernel_size) is int:
            use_large_impl = kernel_size > 5
        else:
            assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
            use_large_impl = kernel_size[0] > 5
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias)

    def refined_downsample(self, dim, downsampling_num):
        block = nn.Sequential()
        for num in range(downsampling_num):
            block.add_module('linear{}'.format(num), nn.Linear(dim, dim))
            block.add_module('nl{}'.format(num), nn.LayerNorm(dim))
            if num != downsampling_num-1:
                block.add_module('pooling{}'.format(num), nn.AdaptiveMaxPool1d(dim//2))
            dim = copy.deepcopy(dim //2)
        return block
    
    def forward(self,img_feat,skel_feat):
        img_feat = img_feat.permute(0,2,1)

        skel_feat = self.adjuster(skel_feat.permute(0,2,1)).permute(0,2,1)

        skel_feat_cl = skel_feat.clone()
        img_feat_cl = img_feat.clone()

        picked_skel = self.pick_best_skel(skel_feat_cl)
        picked_skel = picked_skel.permute(0,2,1)

        picked_img = self.pick_best_img(img_feat_cl)
        picked_img = picked_img.permute(0,2,1)

        skel_feat = skel_feat.permute(0,2,1)
        img_feat = img_feat.permute(0,2,1)

        attn_skel, attn_skel_tensor = self.attn_skel(skel_feat,picked_skel,picked_skel)
        attn_img, attn_img_tensor = self.attn_img(img_feat,picked_img,picked_img)

        skel_feat += attn_skel
        img_feat += attn_img

        skel_feat = skel_feat.permute(0,2,1)
        skel_feat = skel_feat.permute(0,2,1)
        img_feat = self.upsample(img_feat.permute(0,2,1)).permute(0,2,1)

        fused = torch.cat((skel_feat,img_feat),dim=1)
        fused = self.bn(fused)
        fused_attn = self.transformer_encoder(fused)
        fused_attn = fused_attn.mean(1)
        output = self.fc(fused_attn)

        return output
