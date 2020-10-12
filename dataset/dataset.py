"""

固定取样方式下的数据集
"""

import os
import random
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset as dataset
import numpy as np
on_server = False


class Dataset(dataset):
    def __init__(self, ct_dir, seg_dir):
        self.ct_list = os.listdir(ct_dir)
        self.seg_list = list(map(lambda x: x.replace('volume', 'segmentation'), self.ct_list))

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))



    def __getitem__(self, index):
        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        # min max 归一化
        ct_array = ct_array.astype(np.float32)
        ct_array = ct_array / 200
        print(1)
        print(ct_array.shape)
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)
        print(ct_array.shape)
        print(seg_array.shape)

        return ct_array, seg_array

    def __len__(self):
        return len(self.ct_list)


ct_dir = r'D:\Data\Segmentation of hepatic tumors\LITS17\liver\ct'
seg_dir = r'D:\Data\Segmentation of hepatic tumors\LITS17\liver\seg'

train_fix_ds = Dataset(ct_dir, seg_dir)


def random_crop_3d(img, label, crop_size):

    random_z_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]
    random_x_max = img.shape[2] - crop_size[2]

    if random_x_max < 0 or random_y_max < 0 or random_z_max < 0:
        return None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)
    z_random = random.randint(0, random_z_max)

    crop_img = img[z_random:z_random + crop_size[0], y_random:y_random + crop_size[1], x_random:x_random + crop_size[2]]
    crop_label = label[z_random:z_random + crop_size[0], y_random:y_random + crop_size[1], x_random:x_random + crop_size[2]]
    return crop_img, crop_label
# 测试代码
from torch.utils.data import DataLoader

def main():
    train_dl = DataLoader(train_fix_ds, 1, True, num_workers=1, pin_memory=True) # 这样会出错，dataloader中数据大小size要一致！！！
    for index, (ct, seg) in enumerate(train_dl):
        print(2)
        print(type(ct), type(seg))
        print(index, ct.size(), seg.size())
        print('----------------')
if __name__=='__main__':
    main()
