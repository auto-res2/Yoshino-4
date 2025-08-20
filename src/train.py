import torch
import torch.nn as nn
import numpy as np

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        _, (h_n, _) = self.encoder_lstm(x)
        context = h_n[-1].unsqueeze(1)
        repeated_context = context.repeat(1, seq_len, 1)  
        decoded, _ = self.decoder_lstm(repeated_context)
        out = self.output_layer(decoded)
        return out

def create_autoencoder_model(input_dim=2, hidden_dim=8, num_layers=1):
    """Create and return an LSTM autoencoder model for domain shift detection."""
    model = LSTMAutoencoder(input_dim, hidden_dim, num_layers)
    model.eval()
    return model

def predictive_shift_detector(data_window, model, threshold=0.5):
    """Detect domain shifts using LSTM autoencoder reconstruction error."""
    x = torch.tensor(data_window, dtype=torch.float32).unsqueeze(0)  
    with torch.no_grad():
        reconstruction = model(x)
    mse_loss = nn.MSELoss(reduction='mean')
    loss = mse_loss(reconstruction, x).item()
    alert = loss > threshold
    uncertainty = loss
    return alert, uncertainty
