import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from .network_parts import Down1, Down2, Down3, Down4, Up1, Up2, Up3, Up4


class Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Down Convolution
        self.blk1 = Down1()
        self.blk2 = Down2()
        self.blk3 = Down3()
        self.blk4 = Down4()

        # Up sampling
        self.blk5 = Up1()
        self.blk6 = Up2()
        self.blk7 = Up3()
        self.blk8 = Up4()

    def forward(self, x):
        # Down sample
        x1 = self.blk1(x)
        x2 = self.blk2(x1)
        x3 = self.blk3(x2)
        x4 = self.blk4(x3)
        # Up sample
        x5 = self.blk5(x4)
        x6 = self.blk6(x5)
        x7 = self.blk7(x6)
        x8 = self.blk8(x7)
        return x8
