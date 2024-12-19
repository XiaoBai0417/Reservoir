import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load and Prepare Data
# Assume `X` is a NumPy array of features (e.g., Sentinel-2 bands)
# and `y` is a binary label array where 1 = water, 0 = non-water.

# Example: Randomly generated dataset (replace with actual water data)
np.random.seed(42)
num_samples = 1000
num_features = 6  # Example: Sentinel-2 bands ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
X = np.random.rand(num_samples, num_features)
y = np.random.randint(0, 2, size=num_samples)  # Binary labels: 1 = water, 0 = non-water

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for better training performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 2. Define the Feedforward Neural Network (FNN)
class WaterClassifier(nn.Module):
    def __init__(self, input_size):
        super(WaterClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Layer 1: Input to 64 neurons
        self.fc2 = nn.Linear(64, 32)         # Layer 2: 64 neurons to 32
        self.fc3 = nn.Linear(32, 1)          # Output layer: 32 neurons to 1 (binary classification)
        self.relu = nn.ReLU()                # ReLU activation for hidden layers
        self.sigmoid = nn.Sigmoid()          # Sigmoid activation for binary output

    def forward(self, x):
        x = self.relu(self.fc1(x))           # Pass through first layer
        x = self.relu(self.fc2(x))           # Pass through second layer
        x = self.sigmoid(self.fc3(x))        # Output layer
        return x

# 3. Initialize the Model, Loss Function, and Optimizer
input_size = X_train.shape[1]  # Number of input features
model = WaterClassifier(input_size)

criterion = nn.BCELoss()       # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# 4. Train the Model
epochs = 50
batch_size = 32

for epoch in range(epochs):
    model.train()  # Set model to training mode
    for i in range(0, len(X_train_tensor), batch_size):
        # Batch data
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. Evaluate the Model
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    predictions = model(X_test_tensor)
    predictions = (predictions > 0.5).float()  # Convert probabilities to binary labels
    accuracy = accuracy_score(y_test, predictions.numpy())
    print(f'Accuracy on test set: {accuracy:.4f}')
