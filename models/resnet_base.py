import torch
from torch import nn
import torchvision
import torch.utils.data
import torch.nn.functional as F
from einops import rearrange

from dataset.dataset import IrradianceDataset


class ResBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=False)
        fc_input = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(fc_input, 150)
        self.resnet.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        return self.resnet(x)


if __name__ == "__main__":
    pass

