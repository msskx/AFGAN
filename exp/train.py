import torch
import torch.nn as nn
from torch.utils import data

from dataset.Mydataset import FusionDataset
from model.AE import Autoencoder
from model.D import Discriminator
from model.G import Generator
from model.utils import Downsample, gradient
from utils.eval_one_image import evaluation_one

import glob
from torch.nn import functional as F

# 获取数据
train_irimgs_path = glob.glob('../dataset/train/ir/*.png')
train_viimgs_path = glob.glob('../dataset/train/vi/*.png')
train_ds = FusionDataset(train_irimgs_path, train_viimgs_path)

BATCHSIZE = 64
LAMDA = 7
epsilon = 5.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 训练集随机打乱
train_dl = data.DataLoader(dataset=train_ds,
                           batch_size=BATCHSIZE,
                           shuffle=True)

gen = Generator().to(device)
dis = Discriminator().to(device)
if torch.cuda.device_count() > 1:  # 多卡训练
    gen = nn.DataParallel(gen)  # 就在这里wrap一下，模型就会使用所有的GPU
    dis = nn.DataParallel(dis)  # 就在这里wrap一下，模型就会使用所有的GPU
#定义优化器
d_optimizer = torch.optim.Adam(dis.parameters(), lr=1e-4, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.999))
# 定义损失函数
loss_fn = torch.nn.MSELoss()

D_loss = []
G_loss = []
res = []
for epoch in range(5000):
    D_epoch_loss = 0
    G_epoch_loss = 0
    count = len(train_dl)
    # 训练
    for step, (vi, ir) in enumerate(train_dl):
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

        # vi_gen_l1_loss = torch.mean(torch.abs(gen_output - vi*0.5 - ir*0.5))

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
            D_epoch_loss += disc_loss.item()
            G_epoch_loss += gen_loss.item()


    with torch.no_grad():
        D_epoch_loss /= count
        G_epoch_loss /= count
        D_loss.append(D_epoch_loss)
        G_loss.append(G_epoch_loss)
        # 训练完一个epoch就打印输出
        print("Epoch:", epoch, end=' ')
        print(f'D_epoch_loss:{D_epoch_loss},G_epoch_loss{G_epoch_loss}')
