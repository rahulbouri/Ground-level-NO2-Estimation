import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FeatureEmbeddingNN(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(FeatureEmbeddingNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim*2)
        self.fc2 = nn.Linear(embedding_dim*2, embedding_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

class NO2Model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, fc_out_dim):
        super(NO2Model, self).__init__()
        
        # Define embedding networks for each feature
        self.lst_embedding = FeatureEmbeddingNN(1, embedding_dim)
        self.aai_embedding = FeatureEmbeddingNN(1, embedding_dim)
        self.cloud_fraction_embedding = FeatureEmbeddingNN(1, embedding_dim)
        self.precipitation_embedding = FeatureEmbeddingNN(1, embedding_dim)
        self.tropopause_pressure_embedding = FeatureEmbeddingNN(1, embedding_dim)
        
        # LSTM
        self.lstm = nn.LSTM(embedding_dim * 5, hidden_dim, num_layers, batch_first=True)
        
        # Final FCNN
        self.fc1 = nn.Linear(hidden_dim + 2, fc_out_dim)  # +2 for LAT and LON
        self.fc2 = nn.Linear(fc_out_dim, 1)  # Output layer
    
    def forward(self, lst_seq, aai_seq, cloud_fraction_seq, precipitation_seq, tropopause_pressure_seq, lat, lon):
        # Apply embeddings
        lst_embedded = self.lst_embedding(lst_seq)
        aai_embedded = self.aai_embedding(aai_seq)
        cloud_fraction_embedded = self.cloud_fraction_embedding(cloud_fraction_seq)
        precip_embedded = self.precipitation_embedding(precipitation_seq)
        tropopause_pressure_embedded = self.tropopause_pressure_embedding(tropopause_pressure_seq)
        
        # Concatenate embeddings
        embedded_seq = torch.cat((lst_embedded, aai_embedded, cloud_fraction_embedded, precip_embedded, tropopause_pressure_embedded), dim=-1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded_seq)
        lstm_out = lstm_out[:, -1, :]  # Take the output from the last time step
        
        # Concatenate LSTM output with LAT and LON
        lat_lon = torch.cat((lat.unsqueeze(1), lon.unsqueeze(1)), dim=1)
        combined = torch.cat((lstm_out, lat_lon), dim=1)
        
        x = torch.relu(self.fc1(combined))
        out = self.fc2(x)
        
        return out
    

class RMSLoss(nn.Module):
    def __init__(self):
        super(RMSLoss, self).__init__()
    
    def forward(self, predictions, targets):
        # Compute Mean Squared Error (MSE)
        mse = torch.mean((predictions - targets) ** 2)
        # Return the square root of MSE
        rms = torch.sqrt(mse)
        return rms
    
# Model parameters
embedding_dim = 16  # Dimension for feature embeddings
hidden_dim = 64
num_layers = 2
fc_out_dim = 32

# Instantiate model, loss function, and optimizer
model = NO2Model(embedding_dim, hidden_dim, num_layers, fc_out_dim)
criterion = RMSLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
