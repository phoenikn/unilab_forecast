import torch
import torch.nn
import pandas as pd
import numpy as np
from utils.config import *
from einops import rearrange

from torch.utils.data import Dataset


class IrradianceDataset(Dataset):

    def __init__(self, input_len=INPUT_LEN, output_len=OUTPUT_LEN, slip_step=SLIP_STEP, last=False):
        self.input_len = input_len
        self.output_len = output_len
        self.slip_step = slip_step
        self.last = last

        self.irradiance = pd.read_csv("../data/sunshine.csv").fillna(0)["Radiation"]
        self.temperature = pd.read_csv("../data/temp.csv").fillna(0)["Temp"]
        self.hour = pd.read_csv("../data/temp.csv").fillna(0)["Hour"]
        self.hour = (self.hour - self.hour.min()) / (self.hour.max() - self.hour.min())
        self.win_dir = pd.read_csv("../data/wind.csv").fillna(0)["Dir"]
        self.win_dir = (self.win_dir - self.win_dir.min()) / (self.win_dir.max() - self.win_dir.min())
        self.win_spd = pd.read_csv("../data/wind.csv").fillna(0)["Spd"]

    def __len__(self):
        return 1 if self.last else ((len(self.irradiance) // IRRADIANCE_PER_DAY - self.input_len + self.slip_step)
                                    // self.slip_step - self.output_len)

    def _transformer(self, series, width, start, end):
        selected_series = series[start: end].to_numpy()
        selected_series = selected_series.reshape((len(selected_series) // width, width))
        if width == IRRADIANCE_PER_DAY:
            selected_series = np.pad(selected_series, [(0, 0), (5, 4)])
        selected_series = torch.tensor(selected_series)
        return selected_series

    def __getitem__(self, day):
        if self.last:
            day += 290
        start_index_irradiance = day * IRRADIANCE_PER_DAY
        end_index_irradiance = start_index_irradiance + IRRADIANCE_PER_DAY * self.input_len
        day += 10
        target_start = day * IRRADIANCE_PER_DAY
        target_end = target_start + IRRADIANCE_PER_DAY * self.input_len
        start_index_weather = day * WEATHER_PER_DAY
        end_index_weather = start_index_weather + WEATHER_PER_DAY * self.input_len

        selected_irradiance = self._transformer(self.irradiance, IRRADIANCE_PER_DAY,
                                                start_index_irradiance, end_index_irradiance)

        selected_temperature = self._transformer(self.temperature, WEATHER_PER_DAY,
                                                 start_index_weather, end_index_weather)

        selected_dir = self._transformer(self.win_dir, WEATHER_PER_DAY,
                                         start_index_weather, end_index_weather)

        selected_spd = self._transformer(self.win_spd, WEATHER_PER_DAY,
                                         start_index_weather, end_index_weather)

        selected_hour = self._transformer(self.hour, WEATHER_PER_DAY,
                                          start_index_weather, end_index_weather)

        selected_data = torch.stack((selected_irradiance, selected_hour, selected_temperature, selected_dir,
                                     selected_spd))
        selected_data = rearrange(selected_data, "feature day hour -> (day hour) feature")
        target = self.irradiance[target_start: target_end].to_numpy()
        if len(target) < 150 and not self.last:
            raise Exception("Out of range")

        return selected_data, torch.tensor(target)


if __name__ == "__main__":
    dataset = IrradianceDataset()
    print(dataset.__getitem__(0)[0])
    print(dataset.__getitem__(0)[1])
