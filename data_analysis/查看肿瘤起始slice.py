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

ct_dir = 'D:/Data/Segmentation of hepatic tumors/LITS17/tumor/have_tumor/ct'
seg_dir = 'D:/Data/Segmentation of hepatic tumors/LITS17/tumor/have_tumor/seg'
# record_file = './Record/'
# # if os.path.exists(record_file):
# #     shutil.rmtree(record_file)
# begin_end = './Record/begin_end_tumor'
# os.mkdir(record_file)
# os.mkdir(begin_end + '.txt')
# f = open(begin_end,'a+')
tumor = []
for ct_file in os.listdir(ct_dir):

    data = 0
    # 将CT和金标准读入内存
    ct = sitk.ReadImage(os.path.join(ct_dir, ct_file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(seg_dir, ct_file.replace('volume', 'segmentation')), sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    print('*'*10)
    print('effective shape:', ct_array.shape, ',', seg_array.shape)
    # 将金标准中肝脏和背景的标签融合为一个
    seg_array[seg_array <= 1] = 0
    # 调整金标准中肿瘤的标签
    seg_array[seg_array >= 2] = 1

    # 找到肝脏区域开始和结束的slice，并各向外扩张   有的图像没有肿瘤，也就是没有2，会出现超出边界错误
    try:
        z = np.any(seg_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]
    except:
        print('{} have not  toumor'.format(ct_file))
        continue
    # if start_slice > end_slice:
    #     data = start_slice
    #     start_slice = end_slice
    #     end_slice = data
    # tumor.append(end_slice - start_slice)
    print('start:{} end: {} '.format(start_slice, end_slice))

    print(end_slice - start_slice)
    # f.write(str(ct_file)+' '*10+str(start_slice)+' '*5+str(end_slice)+' '*5+str(end_slice - start_slice))
print(tumor)
# f.close()

