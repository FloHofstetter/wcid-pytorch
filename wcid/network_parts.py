import torch.nn as nn
import torch.nn.functional as F

class WCID(nn.Module):
    def __init__(self, in_channels=3, n_classes=1):
        super(WCID, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        # Convolution
        self.blk1 = Down1(in_channels)
        self.blk2 = Down2()
        self.blk3 = Down3()
        self.blk4 = Down4()

        # Convolution Transpose
        self.blk5 = Up1()
        self.blk6 = Up2()
        self.blk7 = Up3()
        self.blk8 = Up4(n_classes)

    def forward(self, x):
        x1 = self.blk1(x)
        x2 = self.blk2(x1)
        x3 = self.blk3(x2)
        x4 = self.blk4(x3)

        x5 = self.blk5(x4)
        x6 = self.blk6(x5)
        x7 = self.blk7(x6)
        x8 = self.blk8(x7)
        return x8


class Down1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.l1 = nn.BatchNorm2d(in_channels)
        self.l2 = nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1))
        self.l3 = nn.ReLU(inplace=True)
        self.l4 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
        self.l5 = nn.ReLU(inplace=True)
        self.l6 = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        return x


class Down2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
        self.l2 = nn.ReLU(inplace=True)
        self.l3 = nn.Dropout2d(0.2)

        self.l4 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        self.l5 = nn.ReLU(inplace=True)
        self.l6 = nn.Dropout2d(0.2)

        self.l7 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1))
        self.l8 = nn.ReLU(inplace=True)
        self.l9 = nn.Dropout2d(0.2)

        self.l10 = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        return x


class Down3(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        self.l2 = nn.ReLU(inplace=True)
        self.l3 = nn.Dropout2d(0.2)

        self.l4 = nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1))
        self.l5 = nn.ReLU(inplace=True)
        self.l6 = nn.Dropout2d(0.2)

        self.l7 = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        return x


class Down4(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.l2 = nn.ReLU(inplace=True)
        self.l3 = nn.Dropout2d(0.2)

        self.l4 = nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1))
        self.l5 = nn.ReLU(inplace=True)
        self.l6 = nn.Dropout2d(0.2)

        self.l7 = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        return x


class Up1(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)

        self.l2 = nn.ConvTranspose2d(64, 64, kernel_size=(5, 5), stride=(1, 1))
        self.l3 = nn.ReLU(inplace=True)
        self.l4 = nn.Dropout2d(0.2)

        self.l5 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.l6 = nn.ReLU(inplace=True)
        self.l7 = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        return x


class Up2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)

        self.l2 = nn.ConvTranspose2d(64, 64, kernel_size=(5, 5), stride=(1, 1))
        self.l3 = nn.ReLU(inplace=True)
        self.l4 = nn.Dropout2d(0.2)

        self.l5 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
        self.l6 = nn.ReLU(inplace=True)
        self.l7 = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        return x


class Up3(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)

        self.l2 = nn.ConvTranspose2d(32, 32, kernel_size=(5, 5), stride=(1, 1))
        self.l3 = nn.ReLU(inplace=True)
        self.l4 = nn.Dropout2d(0.2)

        self.l5 = nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        self.l6 = nn.ReLU(inplace=True)
        self.l7 = nn.Dropout2d(0.2)

        self.l8 = nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), stride=(1, 1))
        self.l9 = nn.ReLU(inplace=True)
        self.l10 = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        return x


class Up4(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.l1 = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)

        self.l2 = nn.ConvTranspose2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
        self.l3 = nn.ReLU(inplace=True)
        self.l4 = nn.ConvTranspose2d(16, n_classes, kernel_size=(3, 3), stride=(1, 1))

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x
