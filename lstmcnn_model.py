import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size=6, num_classes=2):  # Adjust num_classes as needed
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)  # (batch, channels, new_seq_len)
        x = x.permute(0, 2, 1)  # back to (batch, seq_len, channels)
        output, (h_n, c_n) = self.lstm(x)
        out = self.classifier(h_n[-1])
        return out
