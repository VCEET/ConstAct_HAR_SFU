# ConstScene_HAR_SFU
Code and related docs for the results of the collaboration with Dr. Mohammad Mahdavian and Prof. Mo Chen from SFU, Canada. 

# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX


- We provide the dependency file of our experimental environment, you can install all dependencies by creating a new anaconda virtual environment and running `pip install -r requirements.txt ` and `pip install -r requirements2.txt `

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Data Preparation

Please follow [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) for data preparation

To be continued ...

Update the data path and batch size.

## Train

For NTU60:

python main\_train\_ntu.py

For NTU120:

python main\_train\ntu120.py

For ConsAct:

python main\_train\_volvo.py


## Load Pretrained

python main\_train\_volvo.py --load_pretrained True --pretrained_address path/to/address
