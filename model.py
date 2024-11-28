import torch
import torch.nn as nn
import torch.nn.functional as F

class ATTLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.feature_projection = nn.Linear(input_size, hidden_size)
        
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.time_attention = TimeSeriesAttention(hidden_size * 2)  
        
        self.feature_attention = FeatureAttention(hidden_size * 2)

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.residual_proj = nn.Linear(input_size, output_size)
        
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x):

        residual = x[:, -1, :]  
        
        x = self.feature_projection(x)
        
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm1(lstm_out)
        
        context, _ = self.time_attention(lstm_out)
        
        context = self.feature_attention(context)
        
        out = self.fc_layers(context)
        
        residual_out = self.residual_proj(residual)
        out = out + residual_out
        
        return out

class TimeSeriesAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, inputs):

        attention_weights = F.softmax(self.attention(inputs), dim=1)
        attended = torch.sum(attention_weights * inputs, dim=1)
        return attended, attention_weights

class FeatureAttention(nn.Module):

    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):

        weights = self.attention(x)
        return x * weights
    
# The model only contains time dimension attention
class TimeAttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.feature_projection = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.time_attention = TimeSeriesAttention(hidden_size * 2)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        self.residual_proj = nn.Linear(input_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        residual = x[:, -1, :]  
        x = self.feature_projection(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        context, _ = self.time_attention(lstm_out)
        out = self.fc_layers(context)
        residual_out = self.residual_proj(residual)
        return out + residual_out

# The model only contains feature dimension attention
class FeatureAttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.feature_projection = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.feature_attention = FeatureAttention(hidden_size * 2)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        self.residual_proj = nn.Linear(input_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        residual = x[:, -1, :]  
        x = self.feature_projection(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        context = lstm_out[:, -1, :] 
        context = self.feature_attention(context)
        out = self.fc_layers(context)
        residual_out = self.residual_proj(residual)
        return out + residual_out