import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

from utils.eval_one_image import evaluation_one


def save_images_from_tensors(ir, vi,fusion_img, epoch_number):
    # 检查是否有必要创建文件夹
    batch_folder = f"batch-{epoch_number}"
    if not os.path.exists("../exp/result/" + batch_folder):
        print("保存图片的文件夹不存在创建一个")
        os.makedirs("../exp/result/" + batch_folder)
    # 提取第一张图片
    image1 = ir[0].cpu().permute(1, 2, 0)
    image2 = vi[0].cpu().permute(1, 2, 0)
    image3 = fusion_img[0].cpu().permute(1, 2, 0)
    # 将张量转换为数组
    image_array1 = np.array(image1.detach().numpy())
    image_array2 = np.array(image2.detach().numpy())
    image_array3 = np.array(image3.detach().numpy())
    # 保存图片
    v_path = os.path.join("../exp/result/" + batch_folder, f"visible_light-{epoch_number}.png")
    i_path = os.path.join("../exp/result/" + batch_folder, f"infrared_light-{epoch_number}.png")
    f_path = os.path.join("../exp/result/" + batch_folder, f"fused_image-{epoch_number}.png")
    # 评测指标
    plt.imsave(fname = v_path,arr=image_array1.squeeze(axis=-1),cmap='gray')
    plt.imsave(fname = i_path,arr=image_array2.squeeze(axis=-1),cmap='gray')
    plt.imsave(fname = f_path,arr=image_array3.squeeze(axis=-1),cmap='gray')
    # 先存再计算
    EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = evaluation_one(i_path, v_path, f_path)
    return EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM
