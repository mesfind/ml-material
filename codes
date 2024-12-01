import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the dataset
df = pd.read_csv('data/weather_forecast.csv')

# Clean the 'DateTime' column by removing malformed entries
df = df[df['DateTime'].str.match(r'\d{4}-\d{2}-\d{2}.*')]

# Convert 'DateTime' column to datetime format, allowing pandas to infer the format
df["DateTime"] = pd.to_datetime(df["DateTime"], errors='coerce')

# Drop rows where the 'DateTime' conversion resulted in NaT (not-a-time)
df.dropna(subset=["DateTime"], inplace=True)

# Reindex the DataFrame before splitting
df.set_index('DateTime', inplace=True)

# Select only important features
features = ['p(mbar)', 'T(degC)', 'VPmax(mbar)', 'VPdef(mbar)', 'sh(g/kg)', 'rho(g/m**3)', 'wv(m/s)', 'wd(deg)']
df = df[features]

# Resample the DataFrame by day and compute the mean for each day
df_daily = df.resample('D').mean()

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

        return X, y[:, -label_width:]


# Initialize the generator
swg = SlidingWindowGenerator(seq_length=30, label_width=3, shift=1, df=df_daily, label_columns=['wv(m/s)'])

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

# Train and test data loading in tensor format
train_size = int(len(y) * 0.7)
test_size = len(y) - train_size

X_train = torch.Tensor(X[0:train_size])
y_train = torch.Tensor(y[0:train_size])

X_test = torch.Tensor(X[train_size:len(X)])
y_test = torch.Tensor(y[train_size:len(y)])

# Move tensors to the configured device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Create TensorDataset instances for training and testing data
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

# Initialize DataLoader objects for both datasets with batch size 256
batch_size = 512
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Adjust output size

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Training the model
num_epochs = 21
learning_rate = 0.01

input_size = X.shape[2]  # feature dimension
hidden_size = 15
num_layers = 10
output_size = y.shape[1]  # match the label width
lstm = LSTM(input_size, hidden_size, num_layers, output_size).to(device)

criterion = torch.nn.MSELoss()  # Mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    epoch_train_loss = 0.0
    epoch_test_loss = 0.0
    num_train_batches = 0
    num_test_batches = 0

    # Train
    lstm.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = lstm(inputs)
        train_loss = criterion(outputs, targets)
        train_loss.backward()
        optimizer.step()
        epoch_train_loss += train_loss.item()
        num_train_batches += 1

    # Test
    lstm.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            test_outputs = lstm(inputs)
            test_loss = criterion(test_outputs, targets)
            epoch_test_loss += test_loss.item()
            num_test_batches += 1
    
    # Average losses over all batches in the epoch
    epoch_train_loss /= num_train_batches
    epoch_test_loss /= num_test_batches
    
    # Store the average losses
    train_losses.append(epoch_train_loss)
    test_losses.append(epoch_test_loss)
    
    if epoch % 5 == 0:
        print(f"Epoch: {epoch}, Train Loss: {epoch_train_loss:.5f}, Test Loss: {epoch_test_loss:.5f}")

# Plot the losses
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Losses')
plt.legend()
plt.show()



import pandas as pd

def sliding_windows(df, n_in=1, n_out=1, dropnan=True):
    """
    Convert time series data into a supervised learning format for LSTM modeling.

    Parameters:
    - data: The input time series data (pandas DataFrame or list of arrays).
    - n_in: Number of lag observations as input (default: 1).
    - n_out: Number of future observations as output (default: 1).
    - dropnan: Whether to drop rows with NaN values (default: True).

    Returns:
    - Pandas DataFrame with columns representing lag and future observations.
    """

    # Determine the number of variables (features) in the data
    data = df.values
    n_vars = 1 if isinstance(data, list) else data.shape[1]

    # Create a DataFrame from the input data
    dff = pd.DataFrame(data)

    # Initialize lists for column names
    cols, names = list(), list()

    # Create columns for lag observations (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # Create columns for future observations (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # Concatenate the columns to create the final DataFrame
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # Optionally drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


