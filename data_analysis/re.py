"""

将自己随机挑选出来的评价集中的金标准转化为只包含肝脏或者肿瘤区域的mask
主要就是为了方便进行分割结果的查看

０：背景
１：肝脏
２：肿瘤
"""
import os
from time import time

import torch
import torch.nn.functional as F

import numpy as np
import xlsxwriter as xw
import SimpleITK as sitk
import scipy.ndimage as ndimage
import skimage.measure as measure
import skimage.morphology as sm

seg_path = 'C:/Users/xihewen\Desktop\log/tumor_pre_seg'
tumor_seg_path = 'C:/Users/xihewen/Desktop/log/tumor_pre'
threshold = 0.7
for number, file in enumerate(os.listdir(seg_path)):

    seg = sitk.ReadImage(os.path.join(seg_path, file), sitk.sitkInt32)
    seg_array = sitk.GetArrayFromImage(seg)


    # 转换肿瘤
    tumor_array = seg_array.copy()
    tumor_array[seg_array == 1] = 2
    pred_seg = tumor_array
    # 使用线性插值将预测的分割结果缩放到原始nii大小
    pred_seg = torch.FloatTensor(pred_seg).unsqueeze(dim=0).unsqueeze(dim=0)
    pred_seg = F.upsample(pred_seg, seg_array.shape, mode='trilinear').squeeze().detach().numpy()
    pred_seg = (pred_seg > threshold).astype(np.int16)

    # # 先进行腐蚀
    # pred_seg = sm.binary_erosion(pred_seg, sm.ball(5))

    # 取三维最大连通域，移除小区域
    pred_seg = measure.label(pred_seg, 4)
    props = measure.regionprops(pred_seg)

    max_area = 0
    max_index = 0
    for index, prop in enumerate(props, start=1):
        if prop.area > max_area:
            max_area = prop.area
            max_index = index

    pred_seg[pred_seg != max_index] = 0
    pred_seg[pred_seg == max_index] = 1

    pred_seg = pred_seg.astype(np.uint8)

    # # 进行膨胀恢复之前的大小
    # pred_seg = sm.binary_dilation(pred_seg, sm.ball(5))
    # pred_seg = pred_seg.astype(np.uint8)

    print('size of pred: ', pred_seg.shape)
    print('size of GT: ', seg_array.shape)

    tumor_seg = sitk.GetImageFromArray(tumor_array)

    tumor_seg.SetDirection(seg.GetDirection())
    tumor_seg.SetOrigin(seg.GetOrigin())
    tumor_seg.SetSpacing(seg.GetSpacing())
    new_seg_name = 'test-' + 'segmentation-' + str(number) + '.nii'
    sitk.WriteImage(tumor_seg, os.path.join(tumor_seg_path, new_seg_name))

    print(number)
print("over")



