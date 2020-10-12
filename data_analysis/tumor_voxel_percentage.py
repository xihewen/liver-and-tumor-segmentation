"""

查看肿瘤区域像素点个数占据只包含肿瘤区域的slice的百分比
"""
import os

import SimpleITK as sitk


#seg_path = 'D:/Data/Segmentation of hepatic tumors/LITS17/now_begin/train/seg'
# seg_path = "D:/Data/Segmentation of hepatic tumors/LITS17/tumor/size8_noexpand_stride1_thickness2_8256256_8512512/seg"
seg_path ='D:/Data/Segmentation of hepatic tumors/LITS17/tumor/size8_noexpand_stride1_thickness2_8256256_8512512/seg'
total_point = .0
total_liver_point = .0

for index, seg_file in enumerate(os.listdir(seg_path), start=1):

    seg = sitk.ReadImage(os.path.join(seg_path, seg_file))
    seg_array = sitk.GetArrayFromImage(seg)

    liver_slice = 0

    for slice in seg_array:
        if 1 in slice:
            liver_slice += 1

    liver_point = (seg_array > 0).astype(int).sum()

    print('index:{}, precent:{:.4f}'.format(index, liver_point / (liver_slice * 512 * 512) * 100))

    total_point += (liver_slice * 512 * 512)
    total_liver_point += liver_point

print(total_liver_point / total_point)

# 只包含肿瘤的切片中   体素值占比：0.008902068236265053
# 一些expande,  体素值占比：0.010532804645621494
# 全部expande,体素值占比：0.008902068236265053
# 无expend 0.010591231583351669
