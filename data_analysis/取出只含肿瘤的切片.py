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
import scipy.ndimage as ndimage

upper = 200
lower = -200
size = 8  # 取样的slice数量
stride = 1  # 取样的步长
ct_down_scale = 0.5
seg_down_scale = 1
slice_thickness = 2

ct_dir = 'D:/Data/Segmentation of hepatic tumors/LITS17/tumor/have_tumor/ct'
seg_dir = 'D:/Data/Segmentation of hepatic tumors/LITS17/tumor/have_tumor/seg'

new_ct_dir = 'D:/Data/Segmentation of hepatic tumors/LITS17/tumor/only_tumor_slice/ct/'
new_seg_dir = 'D:/Data/Segmentation of hepatic tumors/LITS17/tumor/only_tumor_slice/seg/'

if os.path.exists('D:/Data/Segmentation of hepatic tumors/LITS17/tumor/only_tumor_slice/'):
    shutil.rmtree('D:/Data/Segmentation of hepatic tumors/LITS17/tumor/only_tumor_slice/')

os.mkdir('D:/Data/Segmentation of hepatic tumors/LITS17/tumor/only_tumor_slice/')
os.mkdir(new_ct_dir)
os.mkdir(new_seg_dir)

start_time = time()
for ct_file in os.listdir(ct_dir):

    print(ct_file)
    # 将CT和金标准读入内存
    ct = sitk.ReadImage(os.path.join(ct_dir, ct_file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(seg_dir, ct_file.replace('volume', 'segmentation')), sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    print('effective shape:', ct_array.shape, ',', seg_array.shape)
    # 将金标准中肝脏和背景的标签融合为一个
    seg_array[seg_array == 1] = 0
    # 调整金标准中肿瘤的标签
    seg_array[seg_array == 2] = 1
    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

    # 对CT和金标准进行插值，插值之后的array依然是int类型
    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, ct_down_scale, ct_down_scale), order=3)
    seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / slice_thickness, seg_down_scale, seg_down_scale), order=0)

    # 找到肝脏区域开始和结束的slice，并各向外扩张   有的图像没有肿瘤，也就是没有2，会出现超出边界错误
    try:
        z = np.any(seg_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]
    except:
        print('{} have not  toumor'.format(ct_file))
        continue

    ct_array = ct_array[start_slice:end_slice + 1, :, :]
    seg_array = seg_array[start_slice:end_slice + 1, :, :]

    new_ct = sitk.GetImageFromArray(ct_array)
    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing(
        (ct.GetSpacing()[0] * int(1 / ct_down_scale), ct.GetSpacing()[1] * int(1 / ct_down_scale), slice_thickness))

    new_seg = sitk.GetImageFromArray(seg_array)
    new_seg.SetDirection(ct.GetDirection())
    new_seg.SetOrigin(ct.GetOrigin())
    new_seg.SetSpacing((ct.GetSpacing()[0]* int(1 / seg_down_scale), ct.GetSpacing()[1]* int(1 / seg_down_scale), slice_thickness))

    sitk.WriteImage(new_ct, os.path.join(new_ct_dir, ct_file))
    sitk.WriteImage(new_seg, os.path.join(new_seg_dir, ct_file.replace('volume', 'segmentation')))
