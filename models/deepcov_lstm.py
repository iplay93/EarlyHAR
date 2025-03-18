import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConvLSTM(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(DeepConvLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.lstm1 = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):  # x: (batch, time, channels)
        x = x.permute(0, 2, 1)  # → (batch, channels, time)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # → (batch, time, channels)
        lstm_out, _ = self.lstm1(x)
        output = self.fc(lstm_out[:, -1, :])  # last time step output
        return output
