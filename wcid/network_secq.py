import torch
import torch.nn as nn
import torch.nn.functional as F


class NetSeq(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(NetSeq, self).__init__()
        self.net = nn.Sequential(
            # Down 1
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 8, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Down 2
            nn.Conv2d(16, 32, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(32, 32, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Down 3
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Down 4
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Up 1
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.ConvTranspose2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout2d(),
            # Up 2
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.ConvTranspose2d(64, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout2d(),
            # Up 3
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.ConvTranspose2d(64, 32, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), padding=(2, 2)),
            # nn.ConvTranspose2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout2d(),
            # Up 4
            # nn.ConvTranspose2d(16, 16, kernel_size=(2, 2), stride=(2, 2)),
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.ConvTranspose2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
