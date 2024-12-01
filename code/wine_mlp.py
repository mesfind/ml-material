import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/wine-quality-white-and-red.csv')

# Separate features and target variable
X = df.drop('type', axis=1)
y = df['type']

# Convert categorical target variable to numerical values using LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and testing sets
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(trainX)
X_test = scaler.transform(testX)

# Convert target variables to NumPy arrays and reshape
trainY = np.array(trainY).reshape(-1, 1)
testY = np.array(testY).reshape(-1, 1)

# Convert data to PyTorch tensors with the correct data type
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(trainY)  # Use FloatTensor for BCELoss
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(testY)  # Use FloatTensor for BCELoss

# Define the ANN model
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

input_size = X_train_tensor.shape[1]
hidden_size = 64
output_size = 1
model = ANN(input_size, hidden_size, output_size)

# Move the model to the available device
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Move the tensors to the device
X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)
X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data into PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# Evaluation
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predY = model(X_test_tensor)
    predY = np.round(predY.cpu().numpy()).astype(int).reshape(-1)  # Ensure predictions are integers
# Calculate classification metrics
accuracy = np.mean(predY == testY.reshape(-1))
conf_matrix = confusion_matrix(testY, predY)
class_report = classification_report(testY, predY, target_names=le.classes_)
print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)
# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig("fig/wine_quality_confusion_matrix.png")
plt.show()

# Calculate predicted probabilities using sigmoid activation
with torch.no_grad():
    ypred_proba = torch.sigmoid(model(X_test_tensor)).cpu().numpy()

# Calculate ROC AUC score
roc_auc = roc_auc_score(testY, ypred_proba)

# Compute ROC curve
fpr, tpr, _ = roc_curve(testY, ypred_proba)

# Plot ROC curve in a single figure
plt.figure(figsize=(8, 6))

# Plot ROC curve
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()  # Tighten layout to prevent overlap
plt.savefig("fig/wine_quality_roc_auc.png")
plt.show()
