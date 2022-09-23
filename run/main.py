import os
import sys
import datetime

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


def forward_model(device, data, optimizer, model, criterion, is_train=False):
    inputs, targets = [element.to(device) for element in data]
    targets = targets.float()
    inputs = inputs.float()
    model_name = model.__class__.__name__
    if model_name == "ResBase":
        inputs = rearrange(inputs, "batch (day hour) feature -> batch feature day hour", hour=24)

    if is_train:
        optimizer.zero_grad()

    outputs = model(inputs, targets) if model_name == "GRUBase" else model(inputs)
    loss = criterion(outputs, targets)

    if is_train:
        loss.backward()
        optimizer.step()
    if os.name == "nt":
        print(loss.item())
        print(torch.flatten(outputs.data))
        print(torch.flatten(targets))
        print("------------" * 5)
    return loss


def implement():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------Dataset setting-----------------------------------
    dataset = IrradianceDataset()
    all_data_size = len(dataset)
    train_size = int(0.8 * all_data_size)
    validation_size = all_data_size - train_size
    train_set, validation_set = torch.utils.data.random_split(dataset, [train_size, validation_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

    # --------------------------------Model initialize-----------------------------------
    model = GRUBase()
    model_name = model.__class__.__name__

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(device)

    records = pd.DataFrame(data={"epoch": [], "train_loss": [], "val_loss": []})
    print("Start training!")

    best_loss = 100

    for epoch in range(EPOCH):

        model.train()
        running_train_loss = 0.0
        ten_ratio = -1
        for i, data in enumerate(train_loader, 1):
            # -------------------ratio print------------------------------
            ratio = int(i / len(train_loader) * 10)
            if ratio != ten_ratio:
                print("{}%".format(ratio * 10))
                sys.stdout.flush()
                ten_ratio = ratio

            # ------------------train--------------------------------------
            train_loss = forward_model(device=device, data=data, optimizer=optimizer,
                                       model=model, criterion=criterion, is_train=True)
            running_train_loss += train_loss.item()
        avg_train_loss = running_train_loss / i

        model.eval()
        running_val_loss = 0.0
        for i, data in enumerate(validation_loader, 1):
            val_loss = forward_model(device=device, data=data, optimizer=optimizer,
                                     model=model, criterion=criterion, is_train=False)
            running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / i

        print("Epoch: {} / {}".format(epoch + 1, EPOCH))
        print("Average train loss: ", avg_train_loss)
        print("Average validation loss: ", avg_val_loss)
        sys.stdout.flush()

        if avg_val_loss < best_loss:
            torch.save(model.state_dict(), "../checkpoints/"+model_name+"_"+ str(datetime.datetime.now().date())
                       + ".pth")
            best_loss = avg_val_loss

        records = records.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss},
                                 ignore_index=True)

    records.to_csv("record.csv", index=False)
    torch.save(model.state_dict(), "../checkpoints/"+model_name+".pth")
    print("Finish!")

    print("Device is:", device)


if __name__ == "__main__":
    torch.manual_seed(0)
    implement()
