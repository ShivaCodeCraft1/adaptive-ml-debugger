import os
import yaml
from langchain_core.tools import tool

@tool
def suggest_architecture_fix(strategy: str, config_path: str = "./configs/config.yaml") -> str:
    """
    Applies an architectural fix to the model configuration to combat issues like overfitting 
    or underfitting. Note: Architectural changes require a complete restart of the training job.
    
    Args:
        strategy (str): The fix to apply. Options: 'add_dropout', 'increase_capacity', 'decrease_capacity'.
        config_path (str): Path to the configuration file.
        
    Returns:
        str: Status of the architectural update.
    """
    try:
        if not os.path.exists(config_path):
            return f"Configuration file {config_path} not found. Cannot apply architecture fix."

        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

        msg = ""
        if strategy == "add_dropout":
            current_dropout = config.get("dropout_rate", 0.0)
            new_dropout = min(current_dropout + 0.1, 0.5)  # Cap dropout at 0.5
            config["dropout_rate"] = new_dropout
            msg = f"Increased dropout_rate from {current_dropout} to {new_dropout}."
            
        elif strategy == "increase_capacity":
            # Double the hidden sizes
            sizes = config.get("hidden_sizes", [256, 128, 64])
            new_sizes = [s * 2 for s in sizes]
            config["hidden_sizes"] = new_sizes
            msg = f"Increased hidden_sizes to {new_sizes}."
            
        elif strategy == "decrease_capacity":
            # Halve the hidden sizes to reduce overfitting
            sizes = config.get("hidden_sizes", [256, 128, 64])
            new_sizes = [max(s // 2, 16) for s in sizes]
            config["hidden_sizes"] = new_sizes
            msg = f"Decreased hidden_sizes to {new_sizes}."
            
        else:
            return f"Unknown strategy: '{strategy}'. Valid options: 'add_dropout', 'increase_capacity', 'decrease_capacity'."

        # Add a flag indicating the architecture changed and requires a restart
        config["requires_restart"] = True

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)

        return f"Successfully applied architecture fix: {msg} WARNING: This requires restarting the training job from scratch."
    except Exception as e:
        return f"Failed to apply architecture fix: {str(e)}"