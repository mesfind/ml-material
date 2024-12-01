import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def sliding_windows(df, sequence_length=1, target_column_name=None, train_split=0.8):
    """
    Prepares data for LSTM by creating sliding windows for multi-variable time series prediction.

    Parameters:
    df (pd.DataFrame): Input DataFrame with time series data.
    sequence_length (int): Number of time steps in each sequence.
    target_column_name (str): Name of the target column in the DataFrame.
    train_split (float): Proportion of data to be used for training.

    Returns:
    train_x (np.array): Training sequences.
    train_y (np.array): Training labels.
    test_x (np.array): Testing sequences.
    test_y (np.array): Testing labels.
    """
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Initialize sequences and labels
    sequences = []
    labels = []

    # Find the index of the target column
    target_column_index = df.columns.get_loc(target_column_name)

    # Create sequences and corresponding labels
    for i in range(len(scaled_data) - sequence_length):
        seq = scaled_data[i:i + sequence_length]
        label = scaled_data[i + sequence_length][target_column_index]  # Target column index
        sequences.append(seq)
        labels.append(label)

    # Convert to numpy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)

    # Split into train and test sets
    train_size = int(train_split * len(sequences))
    train_x, test_x = sequences[:train_size], sequences[train_size:]
    train_y, test_y = labels[:train_size].reshape(-1,1), labels[train_size:].reshape(-1,1)
    

    return train_x, train_y, test_x, test_y


# Create a sample DataFrame with random data
np.random.seed(0)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
data = np.random.randn(100, 8)  # 100 rows, 8 columns

df_test = pd.DataFrame(data, index=dates, columns=[f'feature_{i}' for i in range(8)])

# Display the test DataFrame
print(df_test.head())

# Test the sliding_windows function
train_x, train_y, test_x, test_y = sliding_windows(df_test, sequence_length=10, target_column_name='feature_7', train_split=0.8)

# Display shapes of the results
print(f'train_x shape: {train_x.shape}')
print(f'train_y shape: {train_y.shape}')
print(f'test_x shape: {test_x.shape}')
print(f'test_y shape: {test_y.shape}')
