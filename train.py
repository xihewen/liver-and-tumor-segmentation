"""

深度监督下的训练脚本
"""
from time import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from loss.Dice_loss import DiceLoss
from net.att_DialResUNet import net
from dataset.dataset_fix import train_fix_ds as train_ds
import datetime
from logger.logger import Logger
import os
import shutil

# 定义超参数

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
cudnn.benchmark = True
Epoch = 200
leaing_rate_base = 1e-4
alpha = 0.33
batch_size = 2
num_workers = 2
pin_memory = True

net = torch.nn.DataParallel(net).cuda()
net.train()

# record_file = './Record/'
# if os.path.exists(record_file):
#     shutil.rmtree(record_file)  # 递归删除文件夹下面的子文件夹
#
# train_record_txt = './Record/train_record'
# val_record_txt = './Record/val_record'
# os.mkdir(record_file)
# os.mkdir(train_record_txt + '.txt')
# os.mkdir(val_record_txt + '.txt')
#
#
# def write_record(record, file_name):
#     f = open(file_name, 'a')
#     f.write(str(record) + "\n")
#     f.close()


# 定义数据加载
train_dl = DataLoader(train_ds, batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

# 定义损失函数
loss_func = DiceLoss()

# 定义优化器
opt = torch.optim.Adam(net.parameters(), lr=leaing_rate_base)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [100])

# 训练网络
start = time()
logger = Logger('./log')
for epoch in range(Epoch):

    loss = 0
    loss1 = 0
    loss2 = 0
    loss3 = 0
    loss4 = 0

    mean_loss = []

    for step, (ct, seg) in enumerate(train_dl):

        ct = ct.cuda()
        seg = seg.cuda()

        outputs = net(ct)
        print('outputs.size:', outputs.size)
        print('outputs.shape:', outputs.shape)

        loss1 = loss_func(outputs[0], seg)
        loss2 = loss_func(outputs[1], seg)
        loss3 = loss_func(outputs[2], seg)
        loss4 = loss_func(outputs[3], seg)

        loss = (loss1 + loss2 + loss3) * alpha + loss4
        mean_loss.append(loss4.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        lr_decay.step()
        if step % 20 is 0:
            print(
                'epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f},loss:{:.3f}, time:{:.3f} min'
                    .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss4.item(),
                            (time() - start) / 60))
        # train_record = (str(epoch) + ' ' * 5 + str(step) + ' ' * 5 + str(round(loss1.item(), 6)) * 5 + str(
        #     round(loss2.item(), 6)) * 5 + str(round(loss3.item(), 6)) * 5 + str(round(loss4.item(), 6)) * 5 + str(
        #     round(loss.item(), 6)))
        # write_record(train_record, train_record_txt)

    logger.scalar_summary('loss1', loss1, epoch)
    logger.scalar_summary('loss2', loss2, epoch)
    logger.scalar_summary('loss3', loss3, epoch)
    logger.scalar_summary('loss4', loss4, epoch)
    logger.scalar_summary('loss', loss, epoch)

    # if epoch % 10 is 0 and epoch is not 0:
    #
    #     # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
    #     torch.save(net.state_dict(), './module/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss.item(), mean_loss))

    if epoch % 15 is 0 and epoch is not 0:
        alpha *= 0.8
    mean_loss = sum(mean_loss) / len(mean_loss)
    if epoch % 10 is 0 and epoch is not 0:
        now = datetime.datetime.now()
        # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
        torch.save(net.state_dict(),
                   './module/net-{}-{}-{}-{}-{}-{:.3f}.pth'.format(epoch, now.month, now.day, Epoch, batch_size, mean_loss.item()))

# 深度监督的系数变化
# 1.000
# 0.800
# 0.640
# 0.512
# 0.410
# 0.328
# 0.262
# 0.210
# 0.168
# 0.134
# 0.107
# 0.086
# 0.069
# 0.055
# 0.044
# 0.035
# 0.028
# 0.023
# 0.018
# 0.014
# 0.012
# 0.009
# 0.007
# 0.006
# 0.005
# 0.004
# 0.003
# 0.002
# 0.002
# 0.002
# 0.001
# 0.001
# 0.001
# 0.001
# 0.001
# 0.000
# 0.000
