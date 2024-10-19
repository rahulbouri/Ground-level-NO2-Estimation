import torch
import torch.nn as nn
import torch.nn.functional as F

import xgboost as xgb
import numpy as np

class AttentionModel(nn.Module):
    def __init__(self, feature_dims=10, time_steps=15, lstm_units=128, num_heads=8):
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
        self.fc2 = nn.Linear(lstm_units, 8)
        self.bn2 = nn.BatchNorm1d(8)            


    def forward(self, features):

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

        output = torch.sigmoid(self.bn2(self.fc2(output))) 
        
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


class XGBoostModel:
    def __init__(self,best_loss, model_save_path="xgboost-weights/best_xgboost_model.json"):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,  # Set n_estimators to 100
            max_depth=5        # Set max_depth to 5
        )
        self.best_loss = best_loss
        self.model_save_path = model_save_path
    
    def fit(self, input_np: np.ndarray, target_np: np.ndarray):
        assert input_np.shape[1] == 8, "Input tensor must have shape (batch_size, 8)"

        self.model.fit(input_np, target_np)

        return None
    
    def inference(self, input_tensor: torch.Tensor):
        self.model.load_model(self.model_save_path)

        assert input_tensor.shape[1] == 8, "Input tensor must have shape (batch_size, 8)"
        
        if input_tensor.is_cuda:
            device = input_tensor.device
            input_np = input_tensor.detach().cpu().numpy()
        else:
            device = input_tensor.device
            input_np = input_tensor.detach().numpy()
        
        predictions = self.model.predict(input_np)
        predictions_tensor = torch.tensor(predictions, dtype=torch.float32, device=device, requires_grad=True)
    
        return predictions_tensor
    
    def save_model(self):
        self.model.save_model(self.model_save_path)


