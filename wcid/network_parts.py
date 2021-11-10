import torch
import torch.nn as nn
import torch.nn.functional as F


class Down1(nn.Module):
    def __init__(self):
        """

        :param in_channels:
        """
        super().__init__()
        self.l1 = nn.BatchNorm2d(3)
        self.l2 = nn.Conv2d(3, 8, kernel_size=(3, 3), padding=(1, 1))
        self.l3 = nn.ReLU(inplace=True)
        self.l4 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1, 1))
        self.l5 = nn.ReLU(inplace=True)
        self.l6 = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        """

        :param x:
        :return:
        """
        # x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        return x


class Down2(nn.Module):
    def __init__(self):
        """

        :param in_channels:
        """
        super().__init__()
        self.l1 = nn.Conv2d(16, 16, kernel_size=(5, 5), padding=(2, 2))
        # self.l1 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.l2 = nn.ReLU(inplace=True)
        self.l3 = nn.Dropout2d()
        self.l4 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.l5 = nn.ReLU(inplace=True)
        self.l6 = nn.Dropout2d()
        self.l7 = nn.Conv2d(32, 32, kernel_size=(5, 5), padding=(2, 2))
        # self.l7 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.l8 = nn.ReLU(inplace=True)
        self.l9 = nn.Dropout2d()
        self.l10 = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        """

        :param x:
        :return:
        """
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
        """

        :param in_channels:
        """
        super().__init__()
        self.l1 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.l2 = nn.ReLU(inplace=True)
        self.l3 = nn.Dropout2d()
        # self.l4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.l4 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2))
        self.l5 = nn.ReLU(inplace=True)
        self.l6 = nn.Dropout2d()
        self.l7 = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        """

        :param x:
        :return:
        """
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
        """

        """
        super().__init__()
        self.l1 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.l2 = nn.ReLU(inplace=True)
        self.l3 = nn.Dropout2d()
        # self.l4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.l4 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2))
        self.l5 = nn.ReLU(inplace=True)
        self.l6 = nn.Dropout2d()
        self.l7 = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        """

        :param x:
        :return:
        """
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
        """

        :param in_channels:
        """
        super().__init__()
        self.l1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # self.l1 = nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.l2 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2))
        # self.l2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.l3 = nn.ReLU(inplace=True)
        self.l4 = nn.Dropout2d()
        self.l5 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.l6 = nn.ReLU(inplace=True)
        self.l7 = nn.Dropout2d()

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.l1(x)  # Upsample
        if (x.shape[2] * 2 + 1) % 2 != 0:
            x = F.pad(x, (0, 0, 1, 0))
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        return x


class Up2(nn.Module):
    def __init__(self):
        """

        :param in_channels:
        """
        super().__init__()
        self.l1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(1, 1))
        self.l2 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2))
        # self.l2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.l3 = nn.ReLU(inplace=True)
        self.l4 = nn.Dropout2d()
        self.l5 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.l6 = nn.ReLU(inplace=True)
        self.l7 = nn.Dropout2d()

    def forward(self, x):
        """

        :param x:
        :return:
        """
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
        """

        :param in_channels:
        """
        super().__init__()
        self.l1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(1, 1))
        self.l2 = nn.Conv2d(64, 32, kernel_size=(5, 5), padding=(2, 2))
        # self.l2 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1))
        self.l3 = nn.ReLU(inplace=True)
        self.l4 = nn.Dropout2d()
        self.l5 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.l6 = nn.ReLU(inplace=True)
        self.l7 = nn.Dropout2d()
        self.l8 = nn.Conv2d(32, 16, kernel_size=(5, 5), padding=(2, 2))
        self.l8 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.l9 = nn.ReLU(inplace=True)
        self.l10 = nn.Dropout2d()

    def forward(self, x):
        """

        :param x:
        :return:
        """
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
    def __init__(self):
        """

        :param in_channels:
        """
        super().__init__()
        # self.l1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.l1 = nn.ConvTranspose2d(16, 16, kernel_size=(2, 2), stride=(2, 2))
        self.l2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.l3 = nn.ReLU(inplace=True)
        self.l4 = nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(1, 1))
        self.l5 = nn.ReLU(inplace=True)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        return x
