from torch import nn

from model.AE import Autoencoder


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ae = Autoencoder()
    def forward(self, vi, ir):
        # feature_fft = self.fft(vi) #[b,3,256,256]可见光的频谱信息
        # feature_edge = self.edge(ir) #[b,3,256,256]红外光的边缘信息
        # feature_unet = self.unet(feature_edge,feature_fft) #[b,3,256,256]利用Unet进行融合
        x = self.ae(vi, ir)
        # x = feature_edge + feature_fft
        # x = self.ae(x)
        return x