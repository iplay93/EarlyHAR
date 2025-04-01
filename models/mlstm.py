import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, t = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1)
        return x * y.expand_as(x)

class MLSTM_FCN(nn.Module):
    def __init__(self, input_channels, num_classes, lstm_hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(input_channels, lstm_hidden_size, batch_first=True)

        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=8, padding=4)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.se = SqueezeExciteBlock(128)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(lstm_hidden_size + 128, num_classes)

    def forward(self, x):  # x: (batch, time, channels)
        # LSTM branch
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Use last hidden state

        # FCN branch
        x_fcn = x.permute(0, 2, 1)  # (B, C, T)
        x_fcn = F.relu(self.conv1(x_fcn))
        x_fcn = F.relu(self.conv2(x_fcn))
        x_fcn = F.relu(self.conv3(x_fcn))
        x_fcn = self.se(x_fcn)
        x_fcn = nn.functional.adaptive_avg_pool1d(x_fcn, 1).squeeze(-1)  # (B, C)

        # Concatenate both branches
        out = torch.cat((lstm_out, x_fcn), dim=1)
        out = self.dropout(out)
        return self.fc(out)
