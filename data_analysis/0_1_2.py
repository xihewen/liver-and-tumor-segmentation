"""

肝脏分割在自己的测试集下的脚本
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
org ='C:/Users/xihewen/Documents/Tencent Files/1094045696/FileRecv/3dlowres_all'
re = 'C:/Users/xihewen/Desktop/log/tumor_pre'
seg = r'D:\Data\Segmentation of hepatic tumors\LITS17\liver\seg\segmentation-0.nii'

# for file_index, file in enumerate(os.listdir(org)):
#
#
#     # 将CT读入内存
#     ct = sitk.ReadImage(os.path.join(org, file), sitk.sitkInt16)
#
#     ct_array = sitk.GetArrayFromImage(ct)
#
#     print('seg_orgin:', np.unique(ct_array))
#     print(len(np.unique(ct_array)))
#     print(np.unique(ct_array))
ct = sitk.ReadImage(seg, sitk.sitkInt16)

ct_array = sitk.GetArrayFromImage(ct)

print('seg_orgin:', np.unique(ct_array))
print(len(np.unique(ct_array)))
print(np.unique(ct_array))
