#python nii 图像读取，转换成CT 值，设置窗宽窗位，保存成png 图像
import numpy as np
import os  # 遍历文件夹
import nibabel as nib  # nii格式一般都会用到这个包
import imageio  # 转换成图像

center = -500  #肺部的窗宽窗位
width = 1500


def nii_to_image(filepath):
    filenames = os.listdir(filepath)  # 读取nii文件夹

    for f in filenames:
        # 开始读取nii文件
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)  # 读取nii
        img_fdata = img.get_fdata()  # api 已完成转换，读出来的即为CT值
        fname = f.replace('.nii.gz', '')  # 去掉nii的后缀名
        img_f_path = os.path.join(filepath, fname)
        # 创建nii对应的图像的文件夹
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)  # 新建文件夹

        # 转换成窗宽窗位
        min = (2 * center - width) / 2.0 + 0.5
        max = (2 * center + width) / 2.0 + 0.5
        dFactor = 255.0 / (max - min)

        # 开始转换为图像
        (x, y, z) = img.shape
        for i in range(z):  # z是图像的序列
            silce = img_fdata[:, :, i]  # 选择哪个方向的切片都可以

            silce = silce - min
            silce = np.trunc(silce * dFactor)
            silce[silce < 0.0] = 0
            silce[silce > 255.0] = 255  # 转换为窗位窗位之后的数据
            maskimg_slice = maskimg_fdata[:, :, i]

            temp = fname + "_" + '{}.png'.format(i)
            imageio.imwrite(os.path.join(img_f_path, temp),
                            silce[int((x - 512) / 2):int((x - 512) / 2) + 512])


if __name__ == '__main__':
    filepath = '  '
    nii_to_image(filepath)