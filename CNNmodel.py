import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,filt,ker):
        super(SRCNNparam3, self).__init__()

        self.conv1 = nn.Conv2d(3, filt[0], kernel_size=ker[0], stride=(1, 1), padding="same")
        self.conv2 = nn.Conv2d(filt[0], filt[1], kernel_size=ker[1], stride=(1, 1), padding="same")
        self.conv3 = nn.Conv2d(filt[1], 3, kernel_size=ker[2], stride=(1, 1), padding="same")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x