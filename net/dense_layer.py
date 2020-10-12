from typing import Optional
from torch.nn import Sequential, BatchNorm3d, ReLU, Conv3d, Dropout3d
from .bottleneck import Bottleneck

class DenseLayer(Sequential):

    def __init__(self, index :int , in_channels: int, out_channels: int,
                  dropout: float = 0.0):
        super(DenseLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_module('norm', BatchNorm3d(num_features=in_channels))
        self.add_module('relu', ReLU(inplace=True))

        self.add_module('conv', Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, dilation=2*index-1))

        if dropout > 0:
            self.add_module('drop', Dropout3d(dropout, inplace=True))
