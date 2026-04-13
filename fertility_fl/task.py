"""
Task definition for loading federated fertility data
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os

def load_partition_data(partition_id, data_dir='data/processed_dp'):
    """
    Load data for a specific client partition
    
    Args:
        partition_id: Client ID (0, 1, 2, ...)
        data_dir: Directory containing processed data
    
    Returns:
        trainloader, valloader, metadata
    """
    client_dir = os.path.join(data_dir, f'client_{partition_id}')
    
    # Load training data
    X_train = np.load(os.path.join(client_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(client_dir, 'y_train.npy'))
    
    # Load validation data
    X_val = np.load(os.path.join(client_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(client_dir, 'y_val.npy'))
    
    # Load metadata
    with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train.astype(int))
    
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val.astype(int))
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create dataloaders
    trainloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    valloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    return trainloader, valloader, metadata

def load_test_data(data_dir='data/processed_dp'):
    """
    Load global test data for server evaluation
    
    Returns:
        testloader, metadata
    """
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Load metadata
    with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    # Convert to PyTorch tensors
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test.astype(int))
    
    # Create dataset and dataloader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    testloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    return testloader, metadata

def get_model_config(data_dir='data/processed_dp'):
    """
    Get model configuration from metadata
    """
    with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    return {
        'input_dim': metadata['num_features'],
        'num_classes': metadata['num_classes']
    }