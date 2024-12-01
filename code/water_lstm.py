import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

plt.style.use("ggplot")

# Load the dataset
df = pd.read_csv('data/water_table.csv')

# Combine 'Year' and 'Month' into a single datetime column
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1))

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Drop the original 'Year' and 'Month' columns as they are no longer needed
df.drop(columns=['Year', 'Month'], inplace=True)

# Select only important features
features = ['Irrigation', 'Rainfall', 'Tem', 'Evaporation', 'Depth']
# df = df[features]

class SlidingWindowGenerator:
    def __init__(self, seq_length, label_width, shift, df, label_columns=None, dropnan=True):
        self.df = df
        self.label_columns = label_columns
        self.dropnan = dropnan

        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(df.columns)}

        self.seq_length = seq_length
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = seq_length + shift

        self.input_slice = slice(0, seq_length)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def sliding_windows(self):
        data = self.df.values
        X, y = [], []

        for i in range(len(data) - self.total_window_size + 1):
            input_window = data[i:i + self.seq_length]
            label_window = data[i + self.seq_length:i + self.total_window_size]

            # Check for nan values in input_window and label_window by flattening them
            if np.isnan(input_window.flatten()).any() or np.isnan(label_window.flatten()).any():
                continue  # Skip this window if it contains nan values

            X.append(input_window)

            if self.label_columns is not None:
                label_window = label_window[:, [self.column_indices[name] for name in self.label_columns]]
            y.append(label_window)

        X, y = np.array(X), np.array(y)

        return X, y.reshape(-1, self.label_width)


# Initialize the generator
# if label_width=1 it will be single-step forecasting
swg = SlidingWindowGenerator(seq_length=30, label_width=1, shift=1, df=df, label_columns=['Depth'])
print(swg)
# Generate windows
X, y = swg.sliding_windows()

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_shape = X.shape
y_shape = y.shape

X_flat = X.reshape(-1, X_shape[-1])
y_flat = y.reshape(-1, y_shape[-1])

X = scaler_X.fit_transform(X_flat).reshape(X_shape)
y = scaler_y.fit_transform(y_flat).reshape(y_shape)

# Verify the shapes before converting to tensors
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Convert data to PyTorch tensors
X_tensor = torch.Tensor(X)
y_tensor = torch.Tensor(y)

# Verify the tensor shapes
assert X_tensor.size(0) == y_tensor.size(0), "Size mismatch between tensors"

# Move data to GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda:0')  # Use CUDA GPU 0
    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)
else:
    device = torch.device('cpu')

# Define batch size
batch_size = 64

# Create TensorDataset instances for training and testing data
dataset = TensorDataset(X_tensor, y_tensor)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Initialize DataLoader objects for both datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Check for GPU availability including CUDA and Apple's MPS GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda' and torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device = torch.device('cuda:0')  # Use CUDA GPU 0

# Initialize the model
input_size = X.shape[2]  # feature feature
hidden_size = 5
num_layers = 4
output_size = 1
lstm = LSTM(input_size, hidden_size, num_layers, output_size).to(device)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()    # Mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)

# Training the model
num_epochs = 300

train_losses = []
test_losses = []
for epoch in range(num_epochs):
    # Train
    lstm.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs= lstm(inputs)
        train_loss = criterion(outputs, targets)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())
    
    # Test
    lstm.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            test_outputs = lstm(inputs)
            test_loss = criterion(test_outputs, targets)
            test_losses.append(test_loss.item())
    
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Train Loss: {np.mean(train_losses[-len(train_loader):]):.5f}, Test Loss: {np.mean(test_losses[-len(test_loader):]):.5f}")

# Compute final MSE and R² for train and test sets
ypred = lstm(X_tensor[train_size:].to(device)).cpu().detach().numpy()

# Inverse transform the predictions
ypred = scaler_y.inverse_transform(ypred.reshape(-1, 1))
ytest =  scaler_y.inverse_transform(y[train_size:].reshape(-1, 1))

# Compute MSE and R2 scores
test_mse = mean_squared_error(ytest, ypred)
test_r2 = r2_score(ytest, ypred)

# Plot the training and testing loss
plt.figure(figsize=(10, 5))
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Loss Over Epochs')
# Add MSE and R² values as annotations
plt.text(0.5, 0.8, f'Test MSE: {test_mse:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.text(0.5, 0.6, f'Test R²: {test_r2:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.tight_layout()
plt.show()
# Plot observed and predicted values for the test dataset
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(test_predict):], scaler_y.inverse_transform(y[-len(test_predict):]), label='Measurement')
plt.plot(df.index[-len(test_predict):], test_predict, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Depth')
plt.title('Observed vs Predicted Water Table')
plt.legend()
plt.show()

