"""

获取固定取样方式下的训练数据
首先将灰度值超过upper和低于lower的灰度进行截断
然后调整slice thickness，然后将slice的分辨率调整为256*256
只有包含肝脏以及肝脏上下 expand_slice 张slice作为训练样本
最后将输入数据分块，以轴向 stride 张slice为步长进行取样

网络输入为256*256*size
当前脚本依然对金标准进行了缩小，如果要改变，直接修改第70行就行
"""

import os
import shutil
from time import time
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

upper = 200
lower = -200
ct_dir = 'D:/Data/Segmentation of hepatic tumors/LITS17/now_begin/train/ct'
seg_dir = 'D:/Data/Segmentation of hepatic tumors/LITS17/now_begin/train/seg'
new_ct_dir = 'C:/Users/xihewen/Desktop/log'


def overlay(volume_hu_to, segmentation_to_clor, segmentation, alpha):
    # Get binary array for places where an ROI lives  获取ROI所在位置的二进制数组
    segbin = np.greater(segmentation, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    # Weighted sum where there's a value to overlay
    overlayed = np.where(
        repeated_segbin,
        np.round(alpha * segmentation_to_clor + (1 - alpha) * volume_hu_to).astype(np.uint8),
        np.round(volume_hu_to).astype(np.uint8)
    )
    return overlayed


def hu_to_grayscale(volume, hu_min, hu_max):
    # Clip at max and min values if specified  按最大值和最小值剪裁（如果指定）
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1 缩放到0到1之间的值
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval) / max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*返回缩放到0-255范围的值，但*不转换为uint8*
    # Repeat three times to make compatible with color overlay   #重复三次，使其与彩色叠加兼容
    im_volume = 255 * im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)


def class_to_color(segmentation, k_color, t_color):
    # initialize output to zeros  将输出初始化为零
    shp = segmentation.shape
    seg_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32)

    # set output to appropriate color at each location  在每个位置将输出设置为适当的颜色
    seg_color[np.equal(segmentation, 1)] = k_color
    seg_color[np.equal(segmentation, 2)] = t_color
    return seg_color


def main():
    index = 0
    # for ct_file in os.listdir(ct_dir):
    #     # 将CT和金标准读入内存
    #     # if index > 6:
    #     #     break
    #     print(ct_file)
    ct_file = "volume-1.nii"
    ct_array = nib.load(os.path.join(ct_dir, ct_file)).get_data()

    seg_array = nib.load(os.path.join(seg_dir, ct_file.replace('volume', 'segmentation'))).get_data()

    ct_hu_to = hu_to_grayscale(ct_array, -512, 512)

    k_color = [255, 0, 0]
    t_color = [255, 0, 0]
    segmentation_to_clor = class_to_color(seg_array, k_color, t_color)

    alpha = 1
    ct_array = overlay(ct_hu_to, segmentation_to_clor, seg_array, alpha)
    plt.imshow(ct_array[ct_array.shape[0] // 2], origin=lower, cmap='Dark2')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()
    plt.imshow(ct_array[:, ct_array.shape[1] // 2, :], origin=lower, cmap='Dark2')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()
    plt.imshow(ct_array[:, :, ct_array.shape[2] // 2], origin=lower, cmap='Dark2')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
