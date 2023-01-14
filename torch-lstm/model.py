import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,  out_size: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, out_size)

    def init_hidden(self, batch_size: int):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, state

    def forward(self, x: int, hidden: int):
        x = torch.transpose(x, 0, 1)
        all_outputs, hidden = self.lstm(x, hidden)
        out = all_outputs[-1]  # We are interested only in the last output
        x = self.fc(out)
        return x, hidden
