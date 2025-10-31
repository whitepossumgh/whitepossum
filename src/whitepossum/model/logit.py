"""
Logistic Regression
This version fixes all gradient tracking issues
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class LogisticRegression(nn.Module):
    """
    Logistic Regression using PyTorch nn.Module.
    This ensures proper gradient tracking.
    """
    
    def __init__(self, n_features, learning_rate=0.01, n_epochs=100, batch_size=256):
        super(LogisticRegression, self).__init__()
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Use nn.Linear for proper gradient tracking
        self.linear = nn.Linear(n_features, 1)
        
        # Initialize with small weights
        nn.init.uniform_(self.linear.weight, -0.01, 0.01)
        nn.init.zeros_(self.linear.bias)
        
        # Optimizer
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # History
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, x):
        """Forward pass - returns logits."""
        return self.linear(x)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        self.eval()
        X = torch.FloatTensor(X)
        with torch.no_grad():
            logits = self.forward(X)
            probs = torch.sigmoid(logits)
        return probs.numpy().flatten()
    
    def predict(self, X, threshold=0.5):
        """Predict binary labels."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).reshape(-1, 1)
        
        has_val = X_val is not None
        if has_val:
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val).reshape(-1, 1)
        
        n_samples = X_train.shape[0]
        n_batches = n_samples // self.batch_size
        
        print(f"Training Logistic Regression...")
        print(f"Samples: {n_samples}, Batches: {n_batches}, Epochs: {self.n_epochs}")
        
        for epoch in range(self.n_epochs):
            self.train()
            
            # Shuffle
            indices = torch.randperm(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            for i in range(n_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                logits = self.forward(X_batch)
                loss = self.criterion(logits, y_batch)
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"\n❌ NaN detected at epoch {epoch+1}, batch {i+1}")
                    return
                
                epoch_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
            
            # Record training loss
            avg_train_loss = epoch_loss / n_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation loss
            if has_val:
                self.eval()
                with torch.no_grad():
                    val_logits = self.forward(X_val)
                    val_loss = self.criterion(val_logits, y_val)
                    self.val_losses.append(val_loss.item())
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                if has_val:
                    print(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
                else:
                    print(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {avg_train_loss:.4f}")
    
    def plot_training_history(self):
        """Plot training curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', linewidth=2)
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()