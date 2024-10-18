import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import mean_squared_error

from model import AttentionModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class NO2Dataset(Dataset):
    def __init__(self, df, max_days=15):
        self.max_days = max_days
        # Keep a global index directly from the DataFrame's index after sorting
        self.data = df.sort_values(by=['LAT', 'LON', 'Date']).reset_index(drop=False)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Create a column for global indices using the reset index
        self.data['global_idx'] = self.data.index
        
        self.locations = self.data.groupby(['LAT', 'LON']).groups
        self.location_keys = list(self.locations.keys())
        # Global sample index refers to row positions in self.data
        self.samples = [(loc, idx) for loc in self.location_keys for idx in self.locations[loc]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get the location and index from self.samples (global index)
        location, global_data_idx = self.samples[idx]

        # Get all data for the location and sort by 'Date'
        loc_data = self.data.loc[self.locations[location]].sort_values(by='Date')

        # Instead of filtering by global_idx again, directly access the row using iloc
        loc_data_row = loc_data.iloc[(global_data_idx - loc_data.index[0])]

        if loc_data_row is None:
            print(f"No data found for global index {global_data_idx} in location {location}")
            return None  # Handle this case appropriately
        
        # Extract the current date from the location data
        current_date = loc_data_row['Date']
        
        # Define date range for the last `max_days` days including current date
        start_date = current_date - pd.DateOffset(days=self.max_days - 1)
        end_date = current_date

        # Get past data for the last `max_days` days
        past_data = loc_data[(loc_data['Date'] >= start_date) & (loc_data['Date'] <= end_date)]

        # Padding if fewer than `max_days` days
        if len(past_data) < self.max_days:
            num_padding_days = self.max_days - len(past_data)
            padding_dates = pd.date_range(end=start_date - pd.DateOffset(days=1), periods=num_padding_days)
            padding_data = pd.DataFrame({
                'Date': padding_dates,
                'LAT': loc_data['LAT'].iloc[0],  # Fill with location LAT
                'LON': loc_data['LON'].iloc[0],  # Fill with location LON
                'LST': 0, 'AAI': 0, 'CloudFraction': 0, 'Precipitation': 0, 'NO2_strat': 0, 
                'NO2_total': 0, 'NO2_trop': 0, 'TropopausePressure': 0, 'GT_NO2': 0,
                'index': -1,  # Use -1 to indicate padding rows
                'global_idx': -1  # Same for global_idx
                })

            # Concatenate padding and past data
            past_data = pd.concat([padding_data, past_data], ignore_index=True)

        # Sort past data again (optional) to ensure order
        past_data = past_data.sort_values(by='Date').reset_index(drop=True)

        gt_value = loc_data[loc_data['Date'] == current_date]['GT_NO2'].values

        # Extract the relevant features and convert to tensors
        features_tensor = torch.tensor(past_data[['LST', 'AAI', 'CloudFraction', 'Precipitation', 
                                                  'NO2_strat', 'NO2_total', 'NO2_trop', 
                                                  'TropopausePressure', 'LAT', 'LON']].values, dtype=torch.float32)
        # lat = torch.tensor(past_data['LAT'].values[0], dtype=torch.float32)  # Only one LAT value
        # lon = torch.tensor(past_data['LON'].values[0], dtype=torch.float32)  # Only one LON value
        gt = torch.tensor(gt_value[0], dtype=torch.float32)
        # Return the feature tensor, lat/lon, and ground truth
        # return features_tensor, lat, lon, gt
        return features_tensor, gt


def collate_fn(batch):
    features, ground_truths = zip(*batch)

    features_padded = pad_sequence(features, batch_first=True)  # (batch_size, max_seq_len, num_features)
    ground_truths_padded = torch.stack(ground_truths, dim=0)  # (batch_size,) 
    # lat_batch = torch.stack(latitudes, dim=0)  # (batch_size,)
    # lon_batch = torch.stack(longitudes, dim=0)  # (batch_size,)

    return features_padded, ground_truths_padded


class RMSLoss(nn.Module):
    def __init__(self):
        super(RMSLoss, self).__init__()

    def forward(self, predictions, targets):
        # Compute Mean Squared Error (MSE)
        mse = torch.mean((predictions - targets) ** 2)
        # Return the square root of MSE
        rms = torch.sqrt(mse)
        return rms
    

def train_one_epoch(epoch_index, model, criterion, optimizer):
    model.train()  
    model.to(device) 
    running_loss = 0.0
    batch_losses = []

    for i, (features_seq, gt) in enumerate(train_loader):
        features_seq, gt = features_seq.to(device), gt.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(features_seq)
        loss = criterion(outputs.squeeze(), gt)

        # if (i+1) % 25 == 0:
        #     print(f"------------Output Batch {i+1}---------------")
        #     print(outputs.squeeze())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f'Epoch {epoch_index}, Batch {i + 1}, Loss: {loss.item():.4f}')

        if (i+1) % 100 == 0:
            # Calculate the average loss over the last 100 batches
            last_loss = running_loss / 100
            batch_losses.append(last_loss)
            running_loss = 0.0

    return batch_losses


def validate_one_epoch(epoch_index, model, criterion):
    model.eval()  
    model.to(device)  
    val_loss = 0.0
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for i, (features_seq, gt) in enumerate(val_loader):
            # Move data to GPU
            features_seq, gt = features_seq.to(device), gt.to(device)

            val_outputs = model(features_seq)
            loss = criterion(val_outputs.squeeze(), gt)
            val_loss += loss.item()

            print(f'Epoch {epoch_index}, Batch {i + 1}, Loss: {loss.item():.4f}')

            # Collect predictions and ground truths for RMSE calculation
            all_preds.append(val_outputs.squeeze().cpu().numpy())
            all_gts.append(gt.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)
    val_rmse = np.sqrt(np.mean((all_preds - all_gts) ** 2))

    print(f'Epoch {epoch_index}, Validation RMSE: {val_rmse:.4f}')

    return val_loss/len(val_loader)


dataset = pd.read_csv('Train_Cleaned_KNN_Filtered.csv')

batch_size = 128

no2_dataset = NO2Dataset(dataset)
train_size = int(0.8 * len(no2_dataset))
val_size = len(no2_dataset) - train_size

train_dataset, val_dataset = random_split(no2_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

model = AttentionModel()

# Load the saved model weights from a local file (replace 'model_weights.pth' with your filename)
# checkpoint_path = 'trained_models(heads_4)/train-runs-22/latest_Att-CNN-LSTM_model_22.pt'
# model.load_state_dict(torch.load(checkpoint_path))
# model = model.to(device)
print('Model loaded from', checkpoint_path)
#####################################################

criterion = RMSLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)


num_epochs = 50
best_val_loss = float('inf')

all_train_losses = []
all_val_losses = []

for epoch in tqdm(range(1, num_epochs + 1)):

    # Train for one epoch
    train_losses = train_one_epoch(epoch, model, criterion, optimizer)
    all_train_losses.extend(train_losses)
    print(f'||||||||||||||||||||||||Epoch {epoch} Training Completed.||||||||||||||||||||||||')

    # Validate after each epoch
    avg_val_loss = validate_one_epoch(epoch, model, criterion)  
    all_val_losses.append(avg_val_loss)
    print(f"Validation Loss for {epoch}: ", avg_val_loss)

    # Save the last model
    torch.save(model.state_dict(), './trained_models(heads_8)/train-runs-22/latest_Att-CNN-LSTM_model_22.pt')
    print(f'Latest Model Saved {epoch}')

    # Save the model if it has the best validation loss so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), './trained_models(heads_8)/train-runs-22/best_Att-CNN-LSTM_model_22.pt')
        print(f'Model saved at epoch {epoch} with validation RMSE {best_val_loss:.4f}')


plt.figure()
plt.plot(all_train_losses, label='Training Loss')
plt.plot(all_val_losses, label='Validation Loss')
plt.xlabel('Batch (x100)')
plt.ylabel('Loss')
plt.title('Training Loss per 100 Batches')
plt.legend()
plt.show()