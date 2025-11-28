import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import onnx

# ============================================================
# TorchNet Neural Network Class
# ============================================================

class TorchNet(nn.Module):
    def __init__(self, input_dim=8, output_dim=1):
        """
        Initialize neural network with two hidden layers:
        - First hidden layer: 4 neurons
        - Second hidden layer: 3 neurons
        - Output layer: 1 neuron (for binary classification)
        """
        super(TorchNet, self).__init__()

        # Two hidden layers: 4 neurons → 3 neurons
        self.model = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, output_dim),
            nn.Sigmoid()  # For binary classification
        )
        
        self.scaler = None
        self.X_tensor = None
        self.y_tensor = None

    def scale_data(self, X, y):
        """Standardize the input data"""
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)

        self.X_tensor = torch.tensor(self.X_scaled, dtype=torch.float32)
        self.y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        
        return self.X_tensor, self.y_tensor
        
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)
    
    def train_model(self, epochs, criterion, optimizer):
        """
        Train the neural network
        
        Args:
            epochs: Number of training epochs
            criterion: Loss function
            optimizer: Optimization algorithm
            
        Returns:
            loss_history: List of loss values for each epoch
        """
        loss_history = []
        
        for epoch in range(epochs):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self(self.X_tensor)
            loss = criterion(outputs, self.y_tensor)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Store loss
            loss_history.append(loss.item())

            # Print progress every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss = {loss.item():.4f}")
            
        return loss_history
    
    def save_onnx(self, filename="torchnet_diabetes.onnx"):
        """Save the model in ONNX format"""
        # Use a single sample as dummy input
        dummy_input = self.X_tensor[:1]
        
        torch.onnx.export(
            self,
            dummy_input,
            filename,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )

        print(f"Model saved as {filename}")