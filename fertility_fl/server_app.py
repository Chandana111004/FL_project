"""
Flower Server Application for Fertility Risk Prediction
"""

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context, ndarrays_to_parameters
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from flwr.common import Metrics

from fertility_fl.task import load_test_data, get_model_config
from fertility_fl.model import get_model, evaluate

import os
import json
from datetime import datetime
training_history = []

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from multiple clients using weighted average
    """
    # Get total number of samples
    total_samples = sum(num_samples for num_samples, _ in metrics)
    
    # Initialize aggregated metrics
    aggregated = {}
    
    # Get all metric keys from first client
    if metrics:
        metric_keys = metrics[0][1].keys()
        
        for key in metric_keys:
            # Weighted sum
            weighted_sum = sum(
                num_samples * m[key] for num_samples, m in metrics
            )
            # Weighted average
            aggregated[key] = weighted_sum / total_samples
    
    return aggregated

def get_initial_parameters():
    """
    Initialize model parameters for the server
    """
    model_config = get_model_config()
    model = get_model(
        input_dim=model_config['input_dim'],
        num_classes=model_config['num_classes']
    )
    
    # Return initial parameters
    return [val.cpu().numpy() for val in model.state_dict().values()]

def server_evaluate(server_round: int, parameters, config):
    """
    Evaluate global model on centralized test set
    """
    # Load test data
    testloader, metadata = load_test_data()
    
    # Create model and set parameters
    model = get_model(
        input_dim=metadata['num_features'],
        num_classes=metadata['num_classes']
    )
    
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    
    # Evaluate
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    test_loss, test_acc, predictions, labels = evaluate(
        model, testloader, criterion, device
    )
    
    print(f"\n[Round {server_round}] Global Test Metrics:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    # Save results
    os.makedirs('results', exist_ok=True)
    training_history.append({
        'round': server_round,
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'timestamp': datetime.now().isoformat()
    })
    with open('results/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    torch.save({
        'round': server_round,
        'model_state_dict': model.state_dict(),
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'metadata': metadata
    }, f'results/model_round_{server_round}.pth')
    best_acc = max(r['test_accuracy'] for r in training_history)
    if test_acc >= best_acc:
        torch.save({
            'round': server_round,
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }, 'results/best_model.pth')
        print(f"  ★ Best model saved! Accuracy: {test_acc:.4f}")
    return test_loss, {
        "test_accuracy": test_acc,
        "test_loss": test_loss
    }

def server_fn(context: Context):
    """
    Factory function to create server components
    """
    # Get configuration
    num_rounds = context.run_config.get("num-server-rounds", 10)
    fraction_fit = context.run_config.get("fraction-fit", 0.8)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 0.5)
    
    # Initialize strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=3,
        min_evaluate_clients=2,
        min_available_clients=3,
        initial_parameters=ndarrays_to_parameters(get_initial_parameters()),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=server_evaluate,
    )
    
    # Create server config
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(
        strategy=strategy,
        config=config
    )

# Create ServerApp
app = ServerApp(server_fn=server_fn)
