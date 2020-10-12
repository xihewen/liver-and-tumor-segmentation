"""

查看肿瘤区域slice占据整体slice的比例
"""
import os

import SimpleITK as sitk

seg_dir = 'D:/Data/Segmentation of hepatic tumors/LITS17/tumor/haveno_tumor/seg'

for index, seg_file in enumerate(os.listdir(seg_dir), start=1):

    seg = sitk.ReadImage(os.path.join(seg_dir, seg_file))
    seg_array = sitk.GetArrayFromImage(seg)

    liver_slice = 0

    for slice in seg_array:
        if 2 in slice:  # 1为肝脏，2为肿瘤，看看seg这个numpy数组里面有多少1和2
            liver_slice += 1
    if liver_slice == 0:
        print('seg_file:{} have no tumor'.format(seg_file))
# seg_file:segmentation-105.nii have no tumor
# seg_file:segmentation-106.nii have no tumor
# seg_file:segmentation-114.nii have no tumor
# seg_file:segmentation-115.nii have no tumor
# seg_file:segmentation-119.nii have no tumor
# seg_file:segmentation-32.nii have no tumor
# seg_file:segmentation-34.nii have no tumor
# seg_file:segmentation-38.nii have no tumor
# seg_file:segmentation-41.nii have no tumor
# seg_file:segmentation-47.nii have no tumor
# seg_file:segmentation-87.nii have no tumor
# seg_file:segmentation-89.nii have no tumor
# seg_file:segmentation-91.nii have no tumor