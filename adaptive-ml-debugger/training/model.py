import torch
import torch.nn as nn
from typing import List, Dict, Any

class ConfigurableMLP(nn.Module):
    """
    A configurable Multi-Layer Perceptron (MLP) designed for testing the 
    Adaptive ML Pipeline Debugger. 
    
    The architecture allows for dynamic injection of dropout, varying hidden 
    dimensions, and different activation functions so the LangGraph agent can 
    suggest and apply architectural fixes (e.g., adding dropout to fix overfitting).
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_sizes: List[int], 
        output_size: int, 
        dropout_rate: float = 0.0,
        activation: str = "relu"
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Select activation function
        if activation.lower() == "relu":
            self.activation_fn = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation.lower() == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif activation.lower() == "leaky_relu":
            self.activation_fn = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Build layers dynamically
        layers = []
        current_in_features = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_in_features, hidden_size))
            layers.append(self.activation_fn)
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            current_in_features = hidden_size
            
        # Output layer
        layers.append(nn.Linear(current_in_features, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initializes weights. Can be manipulated later by the agent 
        tools to simulate or fix exploding/vanishing gradients.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if isinstance(self.activation_fn, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network. Flattens multi-dimensional inputs 
        (like image batches) automatically.
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        return self.network(x)


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create a model from a configuration dictionary.
    """
    return ConfigurableMLP(
        input_size=config.get("input_size", 784),
        hidden_sizes=config.get("hidden_sizes", [256, 128, 64]),
        output_size=config.get("output_size", 10),
        dropout_rate=config.get("dropout_rate", 0.0),
        activation=config.get("activation", "relu")
    )