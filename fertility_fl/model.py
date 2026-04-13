"""
Neural Network Model for Fertility Risk Prediction
4-class: No Risk / Low Risk / High Risk / Critical Risk
"""
import torch
import torch.nn as nn


class FertilityRiskNet(nn.Module):
    """
    Deep Neural Network for fertility risk prediction
    Bigger network for 4-class classification
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32],
                 num_classes=4, dropout=0.3):
        super(FertilityRiskNet, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GroupNorm(num_groups=1, num_channels=hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def get_model(input_dim, num_classes=4):
    """Factory function to create model"""
    return FertilityRiskNet(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64, 32],
        num_classes=num_classes,
        dropout=0.3
    )


def train_one_epoch(model, trainloader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in trainloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    return total_loss / len(trainloader), correct / total


def evaluate(model, testloader, criterion, device):
    """Evaluate model on test data"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in testloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return total_loss / len(testloader), correct / total, all_preds, all_labels
