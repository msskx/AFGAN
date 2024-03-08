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

BATCHSIZE = 64
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

gen = Generator().to(device)
dis = Discriminator().to(device)
if torch.cuda.device_count() > 1:  # 多卡训练
    gen = nn.DataParallel(gen)  # 就在这里wrap一下，模型就会使用所有的GPU
    dis = nn.DataParallel(dis)  # 就在这里wrap一下，模型就会使用所有的GPU
# 定义优化器
d_optimizer = torch.optim.Adam(dis.parameters(), lr=1e-4, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.999))
# 定义损失函数
loss_fn = torch.nn.MSELoss()

D_loss = []
G_loss = []
res = []
for epoch in range(1000):
    train_D_epoch_loss = 0
    train_G_epoch_loss = 0
    # 开始训练
    gen.train()
    dis.train()
    for step, (vi, ir) in enumerate(train_dl):
        train_count = len(train_dl)
        ir = ir.to(device)
        vi = vi.to(device)
        d_optimizer.zero_grad()
        # 输入真实图片，判别器判定为真
        disc_real_output = dis(vi)  # 输入真实的图片
        d_real_loss = loss_fn(disc_real_output, torch.ones_like(disc_real_output, device=device))
        d_real_loss.backward()  # 反向传播
        # 生成图片
        gen_output = gen(vi, ir)  # 生成图片
        vi_disc_gen_output = dis(gen_output.detach())  # 输入生成图像，判断可见光
        vi_d_fake_loss = loss_fn(vi_disc_gen_output, torch.zeros_like(vi_disc_gen_output, device=device))
        vi_d_fake_loss.backward()  # 生成图片进入判别器进行反向传播
        # 判定器的loss由两部分组成
        disc_loss = d_real_loss + vi_d_fake_loss
        # 更新判别器参数
        d_optimizer.step()
        g_optimizer.zero_grad()
        # 将生成的图片放入判别器，要求骗过判别器
        vi_disc_gen_out = dis(gen_output.detach())
        # 得到生成器的损失
        vi_gen_loss_crossentropyloss = loss_fn(vi_disc_gen_out, torch.ones_like(vi_disc_gen_out, device=device))
        vi_gen_l1_loss = torch.mean(torch.square(gen_output - 0.5 * ir - 0.5 * vi)) + epsilon * torch.mean(
            torch.square(
                gradient(gen_output) -
                0.5 * gradient(vi) -
                0.5 * gradient(vi)
            )
        )
        gen_loss = vi_gen_loss_crossentropyloss + LAMDA * (vi_gen_l1_loss)
        gen_loss.backward()
        # 更新生成器梯度
        g_optimizer.step()
        with torch.no_grad():
            train_D_epoch_loss += disc_loss.item()
            train_G_epoch_loss += gen_loss.item()
    with torch.no_grad():
        train_D_epoch_loss /= train_count
        train_G_epoch_loss /= train_count
        D_loss.append(train_D_epoch_loss)
        G_loss.append(train_G_epoch_loss)
        # 训练完一个epoch就打印输出trainLoss
        print("Epoch:", epoch, end=' ')
        print(f'train_D_epoch_loss:{train_D_epoch_loss},train_G_epoch_loss{train_G_epoch_loss}')
    # 开始测试
    if epoch%49==0:
        # 每49轮来一次
        gen.eval()
        for step, (vi, ir) in enumerate(test_dl):
            test_count = len(test_dl)
            ir = ir.to(device)
            vi = vi.to(device)
            fusion_img = gen(vi, ir)  # 生成图片
            # torch.save(gen.state_dict(), 'result/w/' + 'model_checkpoint.pth')
            # 得到每个batch的数据之后对每个batch进行计算指标
            # 这里我觉得应该输出每个轮次的平均值
            EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = save_images_from_tensors(ir, vi,fusion_img,epoch)
            # 这是每个batch内部的事情
        print('EN:', round(EN, 4))
        print('MI:', round(MI, 4))
        print('SF:', round(SF, 4))
        print('AG:', round(AG, 4))
        print('SD:', round(SD, 4))
        print('CC:', round(CC, 4))
        print('SCD:', round(SCD, 4))
        print('VIF:', round(VIF, 4))
        print('MSE:', round(MSE, 4))
        print('PSNR:', round(PSNR, 4))
        print('Qabf:', round(Qabf, 4))
        print('Nabf:', round(Nabf, 4))
        print('SSIM:', round(SSIM, 4))
        print('MS_SSIM:', round(MS_SSIM, 4))
