import os
import sys

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import pandas as pd
from einops import rearrange

from dataset.dataset import IrradianceDataset
from models.resnet_base import ResBase
from models.gru_base import GRUBase
from utils.config import *


def forecast():
    model = ResBase()
    model.load_state_dict(torch.load("../checkpoints/ResBase_best.pth"))
    dataset = IrradianceDataset(last=True)
    last = dataset.__getitem__(0)[0].float()
    last = rearrange(last, "(day hour) feature -> feature day hour", hour=24)[None, :, :, :]
    model.eval()
    with torch.no_grad():
        pred = model(last)
        pred = pred[0, :]
        pred = torch.relu(pred)
        pred = pd.Series(pred, name="Radiation")
        pred.to_csv("../data/sunshine_pred.csv", index=False)


if __name__ == "__main__":
    forecast()
