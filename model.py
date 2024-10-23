import torch
import torch.nn as nn
import torch.nn.functional as F

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
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
        self.fc3 = nn.Linear(8, 1)

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

        output = self.fc3(output)
        
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
    def __init__(self,best_loss = float('inf'), model_save_path="xgboost-weights/best_xgboost_model.json", n_estimators=150, max_depth=7):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=n_estimators,  
            max_depth=max_depth        
        )
        self.best_loss = best_loss
        self.model_save_path = model_save_path
    
    def fit(self, input_np: np.ndarray, target_np: np.ndarray):

        self.model.fit(input_np, target_np)

        return None
    
    def inference(self, input_tensor: torch.Tensor):
        self.model.load_model(self.model_save_path)

        if isinstance(input_tensor, torch.Tensor):
            if input_tensor.is_cuda:
                input_np = input_tensor.detach().cpu().numpy()
            else:
                input_np = input_tensor.detach().numpy()
        else:
            input_np = input_tensor
        
        predictions = self.model.predict(input_np)
        # predictions_tensor = torch.from_numpy(predictions).float()
    
        return predictions
    
    def save_model(self, model = None):
        if model:
            model.save_model(self.model_save_path)
        else:
            self.model.save_model(self.model_save_path)

    
    def gridsearch_exp(self, X_train, y_train, X_test, y_test, param_grid):
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_model.save_model('exp_xgboost_model.json')

        print("Best model parameters:")
        print(f"n_estimators: {best_model.n_estimators}")
        print(f"max_depth: {best_model.max_depth}")
        print(f"learning_rate: {best_model.learning_rate}")
        print(f"subsample: {best_model.subsample}")
        print(f"colsample_bytree: {best_model.colsample_bytree}")
        print(f"gamma: {best_model.gamma}")
        
        y_pred = best_model.predict(X_test)
        
        rmse = np.sqrt(np.mean(np.square(y_test - y_pred)))
        print(f"RMSE: {rmse}")

        return best_model


