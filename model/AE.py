import torch
from torch import nn
from model.vit import ViT


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Define the encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(True)
        )
        # Define the decoder
        self.vit = ViT(
            image_size=256,  # 图像尺寸
            channels=2,
            patch_size=32,  # patch大小
            num_classes=512 * 16 * 16,  # 最终投影类别
            dim=1024,  # 傻傻维度
            depth=6,  # 傻傻深度
            heads=16,  # 多头头数
            mlp_dim=2048,  # 傻傻mlp维度
            dropout=0.1,
            pool='cls',
            emb_dropout=0.1
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, vi, ir):
        # 将红外和可见光塞入vit
        vit_x = self.vit(torch.cat([vi, ir], axis=1))

        x = torch.cat([vi, ir], dim=1)
        # print(f'初始图像：{x.shape}')
        x = self.encoder(x)
        # print(f'编码器图像：{x.shape}')

        x = x + vit_x.view(-1, 512, 16, 16)  # 将编码器和注意力机制的东西融合

        x = self.decoder(x)
        # print(f'解码器图像：{x.shape}')
        return x
