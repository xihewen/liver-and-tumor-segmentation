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
expand_slice = 0  # 轴向上向外扩张的slice数量
size = 8  # 取样的slice数量
stride = 1  # 取样的步长
ct_down_scale = 1
seg_down_scale = 1
slice_thickness = 2

ct_dir = 'E:/qq_data'
seg_dir = 'E:/qq_data'


# 将CT和金标准读入内存
seg = sitk.ReadImage(os.path.join(ct_dir, 'CHGJ007.nii'), sitk.sitkInt8)
seg_array = sitk.GetArrayFromImage(seg)

ct = sitk.ReadImage(os.path.join(seg_dir, 'CHGJ007_0000.nii'), sitk.sitkInt16)
ct_array = sitk.GetArrayFromImage(ct)

# print(np.unique(ct_array))
# print(np.unique(seg_array))
ct = []
for i,index in enumerate(ct_array[seg_array==0]):
    ct.append(index)
print('')
print(min(ct))
print(max(ct))
print('*')
seg = []
for i,index in enumerate(ct_array[seg_array==1]):
    seg.append(index)
print(min(seg))
print(max(seg))
# print(ct_array[seg_array==1])
colors = ['b', 'g']
p1=plt.figure(figsize=(14,6),dpi=600) #第一幅子图,并确定画布大小
# p1.suptitle('loss',fontsize = 14, fontweight='bold')
# plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)  #位置调整
ax1=p1.add_subplot(1,1,1)
ax2 = ax1.twinx()
ax1.hist([ct, seg], bins=100, color=colors)
n, bins, patches = ax1.hist([ct, seg], bins=100)
ax1.cla()  # clear the axis
# plots the histogram data
width = (bins[1] - bins[0]) * 0.4
bins_shifted = bins + width
ax1.bar(bins[:-1], n[0], width, align='edge', color=colors[0])
ax2.bar(bins_shifted[:-1], n[1], width, align='edge', color=colors[1])
# finishes the plot
ax1.set_ylabel("ct hu Count", color=colors[0])
ax2.set_ylabel("seg hu Count", color=colors[1])
ax1.tick_params('y', colors=colors[0])
ax2.tick_params('y', colors=colors[1])
plt.tight_layout()
plt.savefig("./image/1.png",dpi=600)


# new_seg = sitk.GetImageFromArray(ct_array)
#
# new_seg.SetDirection(ct.GetDirection())
# new_seg.SetOrigin(ct.GetOrigin())
# new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], slice_thickness))
#
#
# sitk.WriteImage(new_seg, 'E:/qq_data/2.nii')
#

