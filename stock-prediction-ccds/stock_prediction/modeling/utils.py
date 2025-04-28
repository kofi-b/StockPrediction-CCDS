import torch
import torch.nn as nn
import numpy as np

# Attention mechanism to focus on key parts of the LSTM output
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attn_weights = self.attn(lstm_output)  # Compute attention scores
        attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize scores
        context = torch.sum(lstm_output * attn_weights, dim=1)  # Weighted sum
        return context

# LSTM model for stock prediction
class StockLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers_count: int):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers_count
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers_count, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)  # Double hidden size due to bidirectionality
        self.dropout = nn.Dropout(0.4)  # Regularization
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Output layer for prediction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  # LSTM forward pass
        context = self.attention(out)  # Apply attention
        out = self.dropout(context)  # Apply dropout
        out = self.fc(out)  # Final prediction
        return out

# Function to add noise for data augmentation
def add_noise(X, noise_level=0.01):
    noise = np.random.normal(0, noise_level, X.shape)  # Generate random noise
    return X + noise