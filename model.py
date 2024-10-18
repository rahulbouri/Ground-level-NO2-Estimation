import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, feature_dims=8, time_steps=15, lstm_units=64, num_heads=4):
        super(AttentionModel, self).__init__()
        
        # Conv1D equivalent in PyTorch
        self.conv1d = nn.Conv1d(in_channels=feature_dims, 
                                out_channels=64, 
                                kernel_size=1)
        
        self.dropout = nn.Dropout(0.5)

        self.bilstm = nn.LSTM(input_size=64, 
                              hidden_size=lstm_units, 
                              batch_first=True, 
                              bidirectional=True)
        
        self.attention_block = AttentionBlock(input_dim=2*lstm_units, time_steps=time_steps, single_attention_vector=True, num_heads=num_heads)
        self.fc1 = nn.Linear(lstm_units*2, lstm_units)  # lstm_units * 2 because it's bidirectional
        self.bn1 = nn.BatchNorm1d(lstm_units)
        self.fc2 = nn.Linear(lstm_units+2, 8)
        self.bn2 = nn.BatchNorm1d(8)            
        self.fc3 = nn.Linear(8, 1)

    def forward(self, features, lat, lon):
        x = features.permute(0, 2, 1)  # Rearrange to (batch_size, INPUT_DIMS, TIME_STEPS)
        x = F.relu(self.conv1d(x))
        x = self.dropout(x)
        
        # Rearrange back to (batch_size, time_steps, conv_output_channels)
        x = x.permute(0, 2, 1)
        
        # BiLSTM expects input shape (batch_size, time_steps, features)
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)

        attention_out = self.attention_block(lstm_out)

        att_pooled, _ = torch.max(attention_out, dim=1)

        output = torch.sigmoid(self.bn1(self.fc1(att_pooled)))

        output = torch.cat((output, lat.unsqueeze(1), lon.unsqueeze(1)), dim=1)

        output = torch.sigmoid(self.bn2(self.fc2(output))) # Take the last time step

        output = self.fc3(output) #TDODO: check attention model
        
        return output


class AttentionBlock(nn.Module):
    def __init__(self, input_dim, time_steps, single_attention_vector=False, num_heads=8):
        super(AttentionBlock, self).__init__()
        self.single_attention_vector = single_attention_vector
        
        self.multihead_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        
        self.fc = nn.Linear(input_dim, time_steps)
        
        if single_attention_vector:
            self.attention_fc = nn.Linear(time_steps, input_dim)
        
    def forward(self, x):
        # x.shape: (batch_size, time_steps, input_dim)
        
        # Multi-head attention
        attn_output, _ = self.multihead_attention(x, x, x)
        
        # Compute attention weights
        attention_weights = self.fc(attn_output)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        if self.single_attention_vector:
            # Average across time steps
            attention_weights = attention_weights.mean(dim=1)
            attention_weights = attention_weights.unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Apply attention weights
        attention_weights = attention_weights.permute(0, 2, 1)
        output_attention_mul = torch.bmm(attention_weights, x)

        return output_attention_mul

