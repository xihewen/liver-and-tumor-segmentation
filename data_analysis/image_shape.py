
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

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

upper = 200
lower = -200
expand_slice = 6  # 轴向上向外扩张的slice数量
size = 8  # 取样的slice数量
stride = 1  # 取样的步长
ct_down_scale = 1
seg_down_scale = 1
slice_thickness = 2

ct_dir = 'D:\Data\Segmentation of hepatic tumors\same_size8_slice_expand6_stride1_thickness2_8256256\ct'
seg_dir = 'D:\Data\Segmentation of hepatic tumors\same_size8_slice_expand6_stride1_thickness2_8256256\seg'


for ct_file in os.listdir(ct_dir):

    # 将CT和金标准读入内存
    ct = sitk.ReadImage(os.path.join(ct_dir, ct_file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(seg_dir, ct_file.replace('volume', 'segmentation')), sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    print('effective shape:', ct_array.shape, ',', seg_array.shape)
