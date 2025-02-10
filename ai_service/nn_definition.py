from torch import nn
from torch.nn.functional import softmax


class SentimentLSTMCore(nn.Module):
    def __init__(
        self,
        input_dim: int = 300,
        hidden_dim: int = 256,
        output_dim: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super(SentimentLSTMCore, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)
        return softmax(logits, dim=1)
