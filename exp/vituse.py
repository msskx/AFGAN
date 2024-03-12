import torch

from model.AE import Autoencoder
from model.vit import ViT
from model.vit.cait import CaiT
from model.vit.cross_vit import CrossViT
v = CaiT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 12,             # depth of transformer for patch to patch attention only
    cls_depth = 2,          # depth of cross attention of CLS tokens to patch
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05    # randomly dropout 5% of the layers
)

# v = ViT(
#     image_size = 512, #图像尺寸
#     channels = 2,
#     patch_size = 32, # patch大小
#     num_classes = 512*16*16, # 最终投影类别
#     dim = 1024,   # 傻傻维度
#     depth = 6, # 傻傻深度
#     heads = 16, # 多头头数
#     mlp_dim = 2048, #傻傻mlp维度
#     dropout = 0.1,
#     pool = 'cls',
#     emb_dropout = 0.1
# )








img = torch.randn(64, 1, 256, 256) # 随机搞个图片看看

preds = v(torch.cat([img,img,img],axis=1)) # (1, 1000)
print("="*50)
print(preds.shape)
print("="*50)
aem = Autoencoder()
p2 = aem(img,img)
print(p2.shape)
