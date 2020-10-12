from __future__ import print_function
import torch
import torch.nn as nn


class dila_conv(nn.Module):  # dilation rate 设计成 锯齿状结构，例如 [1, 2, 5, 1, 2, 5]
    def __init__(self, channels):
        super(dila_conv, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, padding=1),
            nn.PReLU(channels),

            nn.Conv3d(channels, channels, 3, 1, padding=2, dilation=2),
            nn.PReLU(channels),

            nn.Conv3d(channels, channels, 3, 1, padding=5, dilation=5),
            nn.PReLU(channels),
        )

    def forward(self, input):
        x = self.down_conv(input)
        return x


class Down(nn.Module):  # dilation rate 设计成 锯齿状结构，例如 [1, 2, 5, 1, 2, 5]
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.dila_conv = dila_conv(in_channels)
        self.down_conv3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 2, 2),
            # nn.BatchNorm3d(out_channels),
            nn.PReLU(out_channels)
        )

    def forward(self, input):
        dila_x = dila_conv(input)
        x = self.down_conv(dila_x)
        return x


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


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):  # in = out  5 4 4 = 128 96 96
        super(Up, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, 2)
        self.Att = Attention_block(F_g=out_channels, F_x=out_channels, F_int=in_channels)  # 自己添加  ???
        self.conv_block = nn.Conv3d(out_channels + out_channels, out_channels, 2, 2)

    def forward(self, x, conv_x):  # x from bridge  conv_x from 对面

        up_x = self.up(x)  # 通道数不变 = in_channels

        att_x = self.Att(g=up_x, x=conv_x)  # 通道数 从down_in_channels > in_channels

        x = torch.cat((up_x, att_x), dim=1)  # up_x = down_in_channels  x = 需要等于 down_in_channels

        x = self.conv_block(x)

        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        in_channels = 1
        out_channels = 1

        self.inc = nn.Conv3d(in_channels, 16, 1)

        # down
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 48)
        self.down3 = Down(48, 64)
        self.down4 = Down(64, 96)

        self.bridge = nn.Conv3d(96, 128, 1, 0)

        # up
        self.up4 = Up(128, 96)
        self.up3 = Up(96, 64)
        self.up2 = Up(64, 48)
        self.up1 = Up(48, 32)

        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)  # 将图片缩放与标签一致大小
        self.class_conv = nn.Conv3d(32, out_channels, 1)  # class_conv  分类卷积的意思

    def forward(self, input):
        x = input

        x = self.inc(x)

        conv1, x = self.down1(x)

        conv2, x = self.down2(x)

        conv3, x = self.down3(x)

        conv4, x = self.down4(x)

        x = self.bridge(x)

        x = self.up4(x, conv4)

        x = self.up3(x, conv3)

        x = self.up2(x, conv2)

        x = self.up1(x, conv1)

        x = self.up(x)
        x = self.class_conv(x)

        x = nn.Sigmoid(x)

        return x


def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)


net = UNet()
net.apply(init)

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

'''
def main():
    torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = UNet(1,[32,48,64,96,128],3,net_mode='3d').to(device)
    x=torch.rand(4,1,64,96,96)
    x=x.to(device)
    model.forward(x)

if __name__ == '__main__':
    main()
'''
