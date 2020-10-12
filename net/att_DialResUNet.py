"""

基础网络脚本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.nn import Module
from net.dense_layer import DenseLayer


class DenseBlock(Module):

    def __init__(self, in_channels: int, out_channels: int, num_layers: int):
        super(DenseBlock, self).__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.out_channels += self.in_channels

        for i in range(num_layers):
            # 增添dense_layer:norm->relu->conv->dropout
            self.add_module(
                f'layer_{i}',
                DenseLayer(index=i, in_channels=in_channels, out_channels=out_channels)
            )

    def forward(self, block_input):
        layer_input = block_input
        # empty tensor (not initialized) + shape=(0,)
        layer_output = block_input.new_empty(0)

        all_outputs = [block_input] if self.concat_input else []
        for layer in self._modules.values():
            layer_input = torch.cat([layer_input, layer_output], dim=1)
            layer_output = layer(layer_input)
            all_outputs.append(layer_output)

        return torch.cat(all_outputs, dim=1)


class Attention_block(nn.Module):  # 通道数 out    F_g=down_in_channels, F_l=down_in_channels, F_int=in_channels
    def __init__(self, F_g, F_x, F_int):  # F_g = F_l, F_int = 下一级
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(  # out = F_int
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(  # out = F_int
            nn.Conv3d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi_int = nn.Sequential(  # out = 1
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)  # 输出通道数 out
        # 上采样的 l 卷积
        x1 = self.W_x(x)  # 输出通道数 out
        # concat + relu
        psi = self.relu(g1 + x1)  # out =  F_int   数值相加 通道数不变  out
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi_int(psi)  # out =  1个向量
        # 返回加权的 x
        return x * psi  # 通道数 out


class DialResUNet(nn.Module):
    """

    共9498260个可训练的参数, 接近九百五十万
    """

    def __init__(self, training):
        super().__init__()

        self.training = training
        # 通过Sequential将网络层和激活函数结合起来，输出激活后的网络节点
        # encoder_stage1

        self.encoder_stage1 = nn.Sequential(
            DenseBlock(in_channels=16, out_channels=16, num_layers=3))

        self.encoder_stage2 = nn.Sequential(
            DenseBlock(in_channels=32, out_channels=32, num_layers=3),
        )

        self.encoder_stage3 = nn.Sequential(  # dila=1,2,4,8,4,2,1   不能大于一的公约数
            DenseBlock(in_channels=64, out_channels=64, num_layers=3),
        )

        self.encoder_stage4 = nn.Sequential(
            DenseBlock(in_channels=128, out_channels=128, num_layers=3),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        self.att1 = Attention_block(128, 64, 64)  # short_range6, long_range3
        self.att2 = Attention_block(64, 32, 32)  # short_range7, long_range2
        self.att3 = Attention_block(32, 16, 16)  # short_range8, long_range1

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(32, 1, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, 1, 1, 1),
            nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 1, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Sigmoid()
        )

    def forward(self, inputs):

        long_range1 = self.encoder_stage1(inputs) + inputs  # x= short_range1=32  long_range1=16
        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1  # x= short_range2 = 64   long_range2=32
        long_range2 = F.dropout(long_range2, 0.3, self.training)
        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2  # x= short_range3  128   long_range3=64 x+y通道数相加
        long_range3 = F.dropout(long_range3, 0.3, self.training)
        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3  # x= short_range4=256   long_range4=128
        long_range4 = F.dropout(long_range4, 0.3, self.training)
        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, 0.3, self.training)
        output1 = self.map1(outputs)
        #  up  att cat conv

        short_range6 = self.up_conv2(outputs)  # g = short_range6  128
        long_range3 = self.att1(short_range6, long_range3)
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, 0.3, self.training)
        output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)  # g = short_range7 64
        long_range2 = self.att2(short_range7, long_range2)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, 0.3, self.training)
        output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)  # g = short_range8 32
        long_range1 = self.att3(short_range8, long_range1)
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8
        output4 = self.map4(outputs)

        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4


def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)


net = DialResUNet(training=True)
net.apply(init)
# print(net)

# # 输出数据维度检查
# net = net.cuda()
# data = torch.randn((1, 1, 16, 160, 160)).cuda()
# res = net(data)
# for item in res:
#     print(item.size())
#
# 计算网络参数
num_parameter = .0
for item in net.modules():

    if isinstance(item, nn.Conv3d) or isinstance(item, nn.ConvTranspose3d):
        num_parameter += (item.weight.size(0) * item.weight.size(1) *
                          item.weight.size(2) * item.weight.size(3) * item.weight.size(4))

        if item.bias is not None:
            num_parameter += item.bias.size(0)

    elif isinstance(item, nn.PReLU):
        num_parameter += item.num_parameters

print(num_parameter)
