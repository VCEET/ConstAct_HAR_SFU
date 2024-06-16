#work_dir: ./work_dir/ntu60/xsub/lst_joint

# wandb
wandb:  False #True 
wandb_name: 'LLM_ConstAct'
#model_name: 'Fuser_BP_PL_middle24_max3_selfAttn'
#model_name: 'ActionClip_vit16_noText2'
model_name: 'Main'

work_dir :  './output/ConstAct/lst_joint/Main'
# feeder
feeder: feeders.feeder_volvo_RGB_pose.Feeder
#feeder: feeders.feeder_ntu.Feeder

train_feeder_args:
  #data_path: /localhome/mmahdavi/Downloads/volvo_dataset/
  data_path: /localscratch/mmahdavi/TransHAR/data/volvo_dataset/ 
  #data_path: data/ntu/NTU60_CS.npz
  #img_src_path : '/localhome/mmahdavi/Mohammad_ws/human_activity_recognition/TransHAR/data/NTU/data/'
  #img_src_path : '/localscratch/mmahdavi/TransHAR/data/NTU/data/'
  split: train
  debug: False
  random_choose: False
  random_shift: False
  random_move: False
  window_size: 16 #64
  normalization: False
  random_rot: True
  p_interval: [0.5, 1]
  vel: False
  bone: False
  img_frame_interval: 1
#  num_segments : 8

test_feeder_args:
  #data_path: /localhome/mmahdavi/Downloads/volvo_dataset/
  data_path: /localscratch/mmahdavi/TransHAR/data/volvo_dataset/
  #data_path: data/ntu/NTU60_CS.npz
  #img_src_path : '/localhome/mmahdavi/Mohammad_ws/human_activity_recognition/TransHAR/data/NTU/data/'
  #img_src_path : '/localscratch/mmahdavi/TransHAR/data/NTU/data/'
  split: test
  window_size: 16 #64
  p_interval: [0.95]
  vel: False
  bone: False
  debug: False
  img_frame_interval : 1
  
## model
model: model.ctrgcn.Model_lst_4part #_same #mix
model_args:
  num_class: 17
  num_point: 25
  num_person: 1
  graph: graph.ntu_rgb_d.Graph
  k: 8
  head: ['ViT-B/16']
  graph_args:
    labeling_mode: 'spatial'

# model: model.model_poseformer.PoseTransformer
# model_args:
#   num_frame: 64
#   num_joints: 25
#   in_chans: 3
#   embed_dim_ratio: 32
#   depth: 4
#   num_heads: 8
#   mlp_ratio: 2
#   qkv_bias: True
#   qk_scale: null
#   drop_rate: 0.0
#   attn_drop_rate: 0.0
#   drop_path_rate: 0.0
#   norm_layer: null
#   num_class: 60
#   num_person: 2
#   head: ['ViT-B/32']

#optim
weight_decay: 0.0005
base_lr: 5.e-6  # 5.e-3 for resnet  #5.e-6 for ActionClip,  5.e-4 for clip
lr_decay_rate: 0.1
step: [35, 45]
#freeze_epoch: 40
warm_up_epoch: 5
prompt_length: 24
class_position: "middle"

# training
device: [0]
batch_size: 2 #16 # 32
test_batch_size: 2 #16 #32
num_epoch: 55
nesterov: True