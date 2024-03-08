import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

from dataset.Mydataset import FusionDataset
from model.D import Discriminator
from model.G import Generator
from model.utils import gradient

import glob
# 设置一波随机种子
import random

from utils.save import save_images_from_tensors

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 获取数据
train_irimgs_path = glob.glob('../dataset/train/ir/*.png')
train_viimgs_path = glob.glob('../dataset/train/vi/*.png')

test_irimgs_path = glob.glob('../dataset/test/MSRS/ir/*.png')
test_viimgs_path = glob.glob('../dataset/test/MSRS/vi/*.png')

train_ds = FusionDataset(train_irimgs_path, train_viimgs_path)
test_ds = FusionDataset(test_irimgs_path, test_viimgs_path)

BATCHSIZE = 8
LAMDA = 7
epsilon = 5.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 训练集随机打乱
train_dl = data.DataLoader(dataset=train_ds,
                           batch_size=BATCHSIZE,
                           shuffle=True)
test_dl = data.DataLoader(dataset=test_ds,
                          batch_size=BATCHSIZE,
                          shuffle=False)

vi_batch, ir_batch = next(iter(test_dl))

save_images_from_tensors(ir_batch, vi_batch,0.5 * vi_batch + 0.5 * ir_batch,9999)
