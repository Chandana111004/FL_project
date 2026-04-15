"""
Flower Client with Differential Privacy
4-class fertility risk prediction with class weights
"""
import secrets
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from opacus import PrivacyEngine

from fertility_fl.task import load_partition_data, get_model_config
from fertility_fl.model import get_model, evaluate
from fertility_fl.security import check_epsilon_budget, audit_log


class FertilityClientDP(NumPyClient):
    """Flower client with Differential Privacy and class weights"""

    def __init__(self, trainloader, valloader, model_config,
                 local_epochs=3, noise_multiplier=1.0, max_grad_norm=1.0,
                 partition_id=0):
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

        # Initialize model
        self.model = get_model(
            input_dim=model_config['input_dim'],
            num_classes=model_config['num_classes']
        ).to(self.device)

        # ── Class weights to fix imbalance ─────────────────────
        # Compute from training data
        all_labels = []
        for _, y_batch in trainloader:
            all_labels.extend(y_batch.numpy())
        all_labels = np.array(all_labels)

        num_classes = model_config['num_classes']
        class_counts = np.bincount(all_labels.astype(int), minlength=num_classes)
        class_counts = np.where(class_counts == 0, 1, class_counts)  # avoid div by 0
        total = len(all_labels)
        # Weight = total / (num_classes * class_count) — rare classes get higher weight
        weights = total / (num_classes * class_counts)
        weights = weights / weights.sum() * num_classes  # normalize
        class_weights = torch.FloatTensor(weights).to(self.device)

        print(f"  Client {partition_id} class weights: {weights.round(2)}")

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Attach privacy engine
        torch.manual_seed(secrets.randbelow(2**32))
        self.privacy_engine = PrivacyEngine(secure_mode=False)
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.trainloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

        audit_log(f"client_{partition_id}", "initialize", "model", success=True)

    def fit(self, parameters, config):
        """Train with differential privacy"""
        self.set_parameters(parameters)

        for epoch in range(self.local_epochs):
            self.model.train()
            for X_batch, y_batch in self.trainloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

            epsilon, should_stop = check_epsilon_budget(self.privacy_engine)
            if should_stop:
                break

        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        train_loss, train_acc = self._evaluate_train()

        return self.get_parameters(), len(self.trainloader.dataset), {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "epsilon": epsilon,
            "delta": 1e-5
        }

    def evaluate(self, parameters, config):
        """Evaluate model"""
        self.set_parameters(parameters)
        val_loss, val_acc, _, _ = evaluate(
            self.model, self.valloader, self.criterion, self.device
        )
        return val_loss, len(self.valloader.dataset), {
            "val_accuracy": val_acc,
            "val_loss": val_loss
        }

    def _evaluate_train(self):
        """Evaluate on training data"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.trainloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        return total_loss / len(self.trainloader), correct / total

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model._module.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model._module.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model._module.load_state_dict(state_dict, strict=True)


def client_fn(context: Context):
    """Factory function for DP client"""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloader, valloader, metadata = load_partition_data(partition_id)
    model_config = get_model_config()

    noise_multiplier = context.run_config.get("noise-multiplier", 1.0)
    max_grad_norm = context.run_config.get("max-grad-norm", 1.0)

    return FertilityClientDP(
        trainloader=trainloader,
        valloader=valloader,
        model_config=model_config,
        local_epochs=5,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        partition_id=partition_id
    ).to_client()


app = ClientApp(client_fn=client_fn)
