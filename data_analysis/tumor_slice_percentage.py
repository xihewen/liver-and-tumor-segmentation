"""

查看肿瘤区域slice占据整体slice的比例
"""
import os

import SimpleITK as sitk

seg_dir = "D:/Data/Segmentation of hepatic tumors/LITS17/tumor/size8_noexpand_stride1_thickness2_8256256_8512512/seg"
total_slice = .0
total_liver_slice = .0

for index, seg_file in enumerate(os.listdir(seg_dir), start=1):

    seg = sitk.ReadImage(os.path.join(seg_dir, seg_file))
    seg_array = sitk.GetArrayFromImage(seg)

    liver_slice = 0

    for slice in seg_array:
        if 1 in slice:  # 1为肝脏，2为肿瘤，看看seg这个numpy数组里面有多少1和2
            liver_slice += 1

    total_slice += seg_array.shape[0]
    total_liver_slice += liver_slice

    print('seg_file:{}, precent:{:.4f}'.format(seg_file, liver_slice / seg_array.shape[0] * 100))

print(total_liver_slice / total_slice)

# 数据集中包含肿瘤的slice整体占比: 0.12261673317643848
# 合并肝脏和背景区域后,取出肿瘤slice, slice整体占比：0.7822181723509503
# seg_file:segmentation-0.nii, precent:21.3333
# seg_file:segmentation-1.nii, precent:14.6341
# seg_file:segmentation-10.nii, precent:16.1677
# seg_file:segmentation-100.nii, precent:34.3066
# seg_file:segmentation-101.nii, precent:28.1113
# seg_file:segmentation-102.nii, precent:6.6470
# seg_file:segmentation-103.nii, precent:9.3704
# seg_file:segmentation-104.nii, precent:15.6210
# seg_file:segmentation-105.nii, precent:0.0000
# seg_file:segmentation-106.nii, precent:0.0000
# seg_file:segmentation-107.nii, precent:9.3385
# seg_file:segmentation-108.nii, precent:23.5981
# seg_file:segmentation-109.nii, precent:17.3280
# seg_file:segmentation-11.nii, precent:11.3734
# seg_file:segmentation-110.nii, precent:15.9314
# seg_file:segmentation-111.nii, precent:12.2208
# seg_file:segmentation-112.nii, precent:3.5952
# seg_file:segmentation-113.nii, precent:16.2679
# seg_file:segmentation-114.nii, precent:0.0000
# seg_file:segmentation-115.nii, precent:0.0000
# seg_file:segmentation-116.nii, precent:14.6476
# seg_file:segmentation-117.nii, precent:29.3062
# seg_file:segmentation-118.nii, precent:20.3747
# seg_file:segmentation-119.nii, precent:0.0000
# seg_file:segmentation-12.nii, precent:1.9780
# seg_file:segmentation-120.nii, precent:12.2642
# seg_file:segmentation-121.nii, precent:3.4557
# seg_file:segmentation-122.nii, precent:15.1659
# seg_file:segmentation-123.nii, precent:18.9815
# seg_file:segmentation-124.nii, precent:13.7592
# seg_file:segmentation-125.nii, precent:2.1951
# seg_file:segmentation-126.nii, precent:3.7406
# seg_file:segmentation-127.nii, precent:0.3040
# seg_file:segmentation-128.nii, precent:22.0183
# seg_file:segmentation-129.nii, precent:66.8639
# seg_file:segmentation-13.nii, precent:10.4132
# seg_file:segmentation-130.nii, precent:31.8910
# seg_file:segmentation-14.nii, precent:2.7211
# seg_file:segmentation-15.nii, precent:2.1239
# seg_file:segmentation-16.nii, precent:17.4165
# seg_file:segmentation-17.nii, precent:6.4165
# seg_file:segmentation-18.nii, precent:5.2071
# seg_file:segmentation-19.nii, precent:8.7751
# seg_file:segmentation-2.nii, precent:5.6093
# seg_file:segmentation-20.nii, precent:3.8328
# seg_file:segmentation-21.nii, precent:15.5606
# seg_file:segmentation-22.nii, precent:4.8583
# seg_file:segmentation-23.nii, precent:8.1841
# seg_file:segmentation-24.nii, precent:2.5362
# seg_file:segmentation-25.nii, precent:1.8303
# seg_file:segmentation-26.nii, precent:12.7246
# seg_file:segmentation-27.nii, precent:16.7247
# seg_file:segmentation-28.nii, precent:57.3643
# seg_file:segmentation-29.nii, precent:12.7907
# seg_file:segmentation-3.nii, precent:2.8090
# seg_file:segmentation-30.nii, precent:9.0000
# seg_file:segmentation-31.nii, precent:40.6593
# seg_file:segmentation-32.nii, precent:0.0000
# seg_file:segmentation-33.nii, precent:57.7778
# seg_file:segmentation-34.nii, precent:0.0000
# seg_file:segmentation-35.nii, precent:23.3871
# seg_file:segmentation-36.nii, precent:27.0270
# seg_file:segmentation-37.nii, precent:39.3443
# seg_file:segmentation-38.nii, precent:0.0000
# seg_file:segmentation-39.nii, precent:29.2308
# seg_file:segmentation-4.nii, precent:29.0131
# seg_file:segmentation-40.nii, precent:44.2623
# seg_file:segmentation-41.nii, precent:0.0000
# seg_file:segmentation-42.nii, precent:5.6000
# seg_file:segmentation-43.nii, precent:9.6774
# seg_file:segmentation-44.nii, precent:30.2521
# seg_file:segmentation-45.nii, precent:14.8649
# seg_file:segmentation-46.nii, precent:26.6129
# seg_file:segmentation-47.nii, precent:0.0000
# seg_file:segmentation-48.nii, precent:11.0656
# seg_file:segmentation-49.nii, precent:9.0551
# seg_file:segmentation-5.nii, precent:2.0484
# seg_file:segmentation-50.nii, precent:2.9167
# seg_file:segmentation-51.nii, precent:15.4185
# seg_file:segmentation-52.nii, precent:8.8608
# seg_file:segmentation-53.nii, precent:9.5238
# seg_file:segmentation-54.nii, precent:4.1667
# seg_file:segmentation-55.nii, precent:3.1250
# seg_file:segmentation-56.nii, precent:27.6151
# seg_file:segmentation-57.nii, precent:6.0109
# seg_file:segmentation-58.nii, precent:8.0189
# seg_file:segmentation-59.nii, precent:1.8519
# seg_file:segmentation-6.nii, precent:17.7606
# seg_file:segmentation-60.nii, precent:7.3770
# seg_file:segmentation-61.nii, precent:5.6995
# seg_file:segmentation-62.nii, precent:6.9149
# seg_file:segmentation-63.nii, precent:6.7308
# seg_file:segmentation-64.nii, precent:16.5217
# seg_file:segmentation-65.nii, precent:1.9493
# seg_file:segmentation-66.nii, precent:8.1395
# seg_file:segmentation-67.nii, precent:3.0303
# seg_file:segmentation-68.nii, precent:3.3835
# seg_file:segmentation-69.nii, precent:4.0816
# seg_file:segmentation-7.nii, precent:15.3420
# seg_file:segmentation-70.nii, precent:15.3153
# seg_file:segmentation-71.nii, precent:21.2766
# seg_file:segmentation-72.nii, precent:18.2796
# seg_file:segmentation-73.nii, precent:3.3058
# seg_file:segmentation-74.nii, precent:9.3458
# seg_file:segmentation-75.nii, precent:13.4831
# seg_file:segmentation-76.nii, precent:35.1190
# seg_file:segmentation-77.nii, precent:8.5106
# seg_file:segmentation-78.nii, precent:18.1818
# seg_file:segmentation-79.nii, precent:12.2449
# seg_file:segmentation-8.nii, precent:14.9723
# seg_file:segmentation-80.nii, precent:13.8249
# seg_file:segmentation-81.nii, precent:4.0816
# seg_file:segmentation-82.nii, precent:11.5607
# seg_file:segmentation-83.nii, precent:0.5741
# seg_file:segmentation-84.nii, precent:30.0136
# seg_file:segmentation-85.nii, precent:5.3968
# seg_file:segmentation-86.nii, precent:7.5734
# seg_file:segmentation-87.nii, precent:0.0000
# seg_file:segmentation-88.nii, precent:12.5771
# seg_file:segmentation-89.nii, precent:0.0000
# seg_file:segmentation-9.nii, precent:14.5719
# seg_file:segmentation-90.nii, precent:21.8375
# seg_file:segmentation-91.nii, precent:0.0000
# seg_file:segmentation-92.nii, precent:5.3828
# seg_file:segmentation-93.nii, precent:28.5920
# seg_file:segmentation-94.nii, precent:12.9771
# seg_file:segmentation-95.nii, precent:1.6647
# seg_file:segmentation-96.nii, precent:29.2244
# seg_file:segmentation-97.nii, precent:28.3159
# seg_file:segmentation-98.nii, precent:21.8605
# seg_file:segmentation-99.nii, precent:14.4674
# 0.12261673317643848


