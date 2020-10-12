import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os


import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

niiSegPath = r'D:\Data\Segmentation of hepatic tumors\LITS17\liver\seg\segmentation-'
niiImagePath = r'D:\Data\Segmentation of hepatic tumors\LITS17\liver\ct\volume-'

def getRangeImageDepth(image):
    z = np.any(image, axis=(1, 2))  # z.shape:(depth,)
    # print("all index:",np.where(z)[0])
    if len(np.where(z)[0]) > 0:
        startposition, endposition = np.where(z)[0][[0, -1]]
    else:
        startposition = endposition = 0
    return startposition, endposition


"""
会画出每个病人肿瘤区域最大切片的直方图
与汇总的直方图
"""
def total_hu(srcimg, segimg):  # 寻找每个病例的全部像素值分类
    seg_liver = segimg.copy()
    seg_liver[seg_liver > 0] = 1

    seg_tumorimage = segimg.copy()
    seg_tumorimage[segimg == 1] = 0
    seg_tumorimage[segimg == 2] = 1

    # max_tumor = 0  # 记录肿瘤的最大占比
    # max_tumor_index = -1  # 记录最大肿瘤所在切片
    liver_hu = []  # liver hu value
    tumor_hu = []
    # 获取含有肿瘤切片的起、止位置 以及像素值
    start_tumor, end_tumor = getRangeImageDepth(seg_tumorimage)
    for j in range(start_tumor, end_tumor + 1):
        src_flatten = srcimg[j].flatten()
        tumor_flatten = seg_tumorimage[j].flatten()
        for j in range(src_flatten.shape[0]):
            if tumor_flatten[j] > 0:
                tumor_hu.append(src_flatten[j])
        # 获取含有肝脏切片的起、止位置 以及像素值

    start_liver, end_liver = getRangeImageDepth(seg_liver)
    for j in range(start_liver, end_liver + 1):
        src_flatten = srcimg[j].flatten()
        liver_flatten = seg_liver[j].flatten()
        for j in range(src_flatten.shape[0]):
            if liver_flatten[j] > 0:
                liver_hu.append(src_flatten[j])
    return liver_hu, tumor_hu
total_liver = []
total_tumor = []
colors = ['b', 'g']
for i in range(0, 131, 1):
    seg = sitk.ReadImage(niiSegPath  + str(i) + ".nii", sitk.sitkUInt8)
    segimg = sitk.GetArrayFromImage(seg)
    src = sitk.ReadImage(niiImagePath  + str(i) + ".nii")
    srcimg = sitk.GetArrayFromImage(src)

    liver,tumor = total_hu(srcimg, segimg)

    total_liver.append(liver)
    total_tumor.append(tumor)
    """    
    # 因为肝脏区域很多而肿瘤很少，所以画直方图使用相同的y-axis就会导致
    # 几乎看不到肿瘤的直方图
    plt.hist(flat_total_liver, color = "skyblue", bins=100, alpha=0.5, 	    label='liver hu')
    plt.hist(flat_total_tumor, color = "red", bins=100, alpha=0.5, label='tumor hu')
    plt.legend(loc='upper right')
    """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.hist([liver, tumor], bins=100, color=colors)
    n, bins, patches = ax1.hist([liver, tumor], bins=100)  #n : array or list of arrays(箱子的值)bins : array(箱子的边界)patches : list or list of lists
    ax1.cla()  # clear the axis

    # plots the histogram data
    width = (bins[1] - bins[0]) * 0.4
    bins_shifted = bins + width
    ax1.bar(bins[:-1], n[0], width, align='edge', color=colors[0])
    ax2.bar(bins_shifted[:-1], n[1], width, align='edge', color=colors[1])

    # finishes the plot
    ax1.set_ylabel("liver hu Count", color=colors[0])
    ax2.set_ylabel("tumor hu Count", color=colors[1])
    ax1.tick_params('y', colors=colors[0])
    ax2.tick_params('y', colors=colors[1])
    plt.tight_layout()
    plt.title("person[%d]" % i)
    plt.savefig("./hist_image/person%d.png"%i,dpi=600)
    # plt.show()
    plt.clf()

# 用来展平total_liver和total_tumor里面的值
flat_total_liver = [item for sublist in total_liver for item in sublist]
flat_total_tumor = [item for sublist in total_tumor for item in sublist]
# plt.hist(flat_total_liver, color = "skyblue", bins=100, alpha=0.5, label='liver hu')
# plt.hist(flat_total_tumor, color = "red", bins=100, alpha=0.5, label='tumor hu')
# plt.legend(loc='upper right')
# plt.title("total hu")
# plt.savefig("LITS/total.png")
# plt.clf()


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.hist([flat_total_liver, flat_total_tumor], bins=100, color=colors)
n, bins, patches = ax1.hist([flat_total_liver, flat_total_tumor], bins=100)
ax1.cla()  # clear the axis

# plots the histogram data
width = (bins[1] - bins[0]) * 0.4
bins_shifted = bins + width
ax1.bar(bins[:-1], n[0], width, align='edge', color=colors[0])
ax2.bar(bins_shifted[:-1], n[1], width, align='edge', color=colors[1])

# finishes the plot
ax1.set_ylabel("liver hu Count", color=colors[0])
ax2.set_ylabel("tumor hu Count", color=colors[1])
ax1.tick_params('y', colors=colors[0])
ax2.tick_params('y', colors=colors[1])
plt.tight_layout()
plt.title("total")
plt.savefig("./hist_image/total.png")
plt.show()
plt.clf()
