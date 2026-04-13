"""
Flower Client Application for Fertility Risk Prediction
"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import torch
import torch.nn as nn
import torch.optim as optim

from fertility_fl.task import load_partition_data, get_model_config
from fertility_fl.model import get_model, train_one_epoch, evaluate

class FertilityClient(NumPyClient):
    """
    Flower client for training fertility risk prediction model
    """
    def __init__(self, trainloader, valloader, model_config, local_epochs=3):
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = get_model(
            input_dim=model_config['input_dim'],
            num_classes=model_config['num_classes']
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def fit(self, parameters, config):
        """
        Train model on local data
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Train for multiple local epochs
        for epoch in range(self.local_epochs):
            train_loss, train_acc = train_one_epoch(
                self.model,
                self.trainloader,
                self.optimizer,
                self.criterion,
                self.device
            )
        
        # Return updated model parameters and metrics
        return self.get_parameters(), len(self.trainloader.dataset), {
            "train_loss": train_loss,
            "train_accuracy": train_acc
        }
    
    def evaluate(self, parameters, config):
        """
        Evaluate model on local validation data
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate
        val_loss, val_acc, _, _ = evaluate(
            self.model,
            self.valloader,
            self.criterion,
            self.device
        )
        
        return val_loss, len(self.valloader.dataset), {
            "val_accuracy": val_acc,
            "val_loss": val_loss
        }
    
    def get_parameters(self):
        """Get model parameters as numpy arrays"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        """Set model parameters from numpy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

def client_fn(context: Context):
    """
    Factory function to create a Flower client
    """
    # Get partition ID from context
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Load data for this partition
    trainloader, valloader, metadata = load_partition_data(partition_id)
    
    # Get model configuration
    model_config = get_model_config()
    
    # Create and return client
    return FertilityClient(
        trainloader=trainloader,
        valloader=valloader,
        model_config=model_config,
        local_epochs=3
    ).to_client()

# Create ClientApp
app = ClientApp(client_fn=client_fn)