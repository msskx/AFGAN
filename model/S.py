# siamese network
from torch import nn
import torch

from model.AE import Encoder


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.encoder = Encoder()
        self.flatten = nn.Flatten()

        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(131072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        print(input.shape)
        vi = input[:,0]
        ir = input[:,1]

        print(f'vi:{vi.shape};ir:{ir.shape}')

        vi_output = self.encoder(vi)
        ir_output = self.encoder(ir)

        vi_output = self.flatten(vi_output)
        ir_output = self.flatten(ir_output)

        combine = vi_output * ir_output

        print(f"combine {combine.shape}")
        output = self.cls_head(combine)
        return output


ins = torch.randn(1, 2, 256, 256)

model = Siamese()
output = model(ins)
print(output.shape)
