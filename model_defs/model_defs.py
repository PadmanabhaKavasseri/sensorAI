import torch
import torch.nn as nn
import torch.nn.functional as F

class GestureRecCNN_V3(nn.Module):
    def __init__(self, input_size=6, num_classes=2, dropout_rate=0.3):
        super().__init__()
        
        # Multi-scale feature extraction
        self.conv_blocks = nn.ModuleList([
            # Short-term patterns (3-sample kernel)
            self._make_conv_block(input_size, 32, 3, 1),
            # Medium-term patterns (7-sample kernel)  
            self._make_conv_block(input_size, 32, 7, 1),
            # Long-term patterns (15-sample kernel)
            self._make_conv_block(input_size, 32, 15, 1),
        ])
        
        # Feature fusion
        self.fusion_conv = nn.Conv1d(96, 128, kernel_size=1)  # 32*3 = 96
        
        # Temporal attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final processing
        self.final_conv = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def _make_conv_block(self, in_channels, out_channels, kernel_size, stride):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        
        # Multi-scale feature extraction
        features = []
        for conv_block in self.conv_blocks:
            feat = conv_block(x)
            # Ensure all features have same temporal dimension
            feat = F.adaptive_avg_pool1d(feat, 100)  # Standardize to 100 time steps
            features.append(feat)
        
        # Concatenate multi-scale features
        x = torch.cat(features, dim=1)  # (batch, 96, 100)
        
        # Feature fusion
        x = self.fusion_conv(x)  # (batch, 128, 100)
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Final processing
        x = self.final_conv(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)
        
        # Classification
        out = self.classifier(x)
        return out

class GestureRecCNN_V2(nn.Module):
    def __init__(self, input_size=6, num_classes=2, dropout_rate=0.2):
        super().__init__()
        
        self.cnn = nn.Sequential(
            # First block
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout1d(dropout_rate),
            
            # Second block
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout1d(dropout_rate),
            
            # Third block
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout1d(dropout_rate),
            
            # Fourth block
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, seq_len, features) → (batch, features, seq_len)
        x = self.cnn(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)
        out = self.classifier(x)
        return out

class GestureRecCNN_V1(nn.Module):
    def __init__(self, input_size=6, num_classes=2):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Reduces seq_len by 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Reduces seq_len by 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), # Reduces to (batch, 128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch, seq_len, features) → (batch, features, seq_len)
        x = self.cnn(x) # (batch, 128, 1)
        x = x.squeeze(-1) # (batch, 128)
        out = self.classifier(x)
        return out

class LSTMGestureModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, num_classes=2, dropout_rate=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        
        # Attention mechanism
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_size*2)
        
        # Classification
        out = self.classifier(context)
        return out

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

