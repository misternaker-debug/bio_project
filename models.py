import torch
import torch.nn as nn

class ExpressionCNN(nn.Module):
    def __init__(self, input_length, n_tissues, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, n_tissues)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        layers = []
        self.flatten = nn.Flatten()
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out)
        return out