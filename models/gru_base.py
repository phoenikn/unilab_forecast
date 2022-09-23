import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
from einops import rearrange

from dataset.dataset import IrradianceDataset


class GRUBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(input_size=5, hidden_size=16, num_layers=4, batch_first=True)
        self.decoder = nn.GRU(input_size=5, hidden_size=16, num_layers=4, batch_first=True)

        self.fc1 = nn.Linear(16, 1)
        self.fc2 = nn.Linear(16, 5)

        # -----------linear decoder---------------
        self.fc = nn.Linear(64, 150)

    def forward(self, input_seq, target_seq):
        _, hidden = self.encoder(input_seq)

        hidden = rearrange(hidden, "l b h -> b (l h)")
        return self.fc(hidden)

        # ------------------------decoder part, needs improvement-------------------
        # decoder_input = input_seq[:, -1:, :]
        #
        # output_list = []
        #
        # for i in range(target_seq.shape[1]):
        #     output, hidden = self.decoder(decoder_input, hidden)
        #     output_list.append(output)
        #     decoder_input = self.fc2(output)
        # output_tensor = torch.cat(output_list, 1)
        # output_tensor = torch.relu(output_tensor)
        # output_tensor = self.fc1(output_tensor)
        # output_tensor = torch.squeeze(output_tensor, 2)
        #
        # return output_tensor


if __name__ == "__main__":
    dataset = IrradianceDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    test_gru = GRUBase()
    input_test = iter(loader).__next__()
    test_gru(input_test[0].float(), input_test[1].float())
