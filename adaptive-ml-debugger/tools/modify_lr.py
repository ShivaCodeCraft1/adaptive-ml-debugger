import os
import yaml
from langchain_core.tools import tool

@tool
def modify_learning_rate(new_lr: float, config_path: str = "./configs/config.yaml") -> str:
    """
    Modifies the learning rate in the configuration file to help stabilize training.
    Useful when loss is diverging or gradients are exploding.
    
    Args:
        new_lr (float): The new learning rate to apply (e.g., 0.0001).
        config_path (str): Path to the configuration file.
        
    Returns:
        str: A success or error message regarding the update.
    """
    try:
        if not os.path.exists(config_path):
            # Create a default directory structure if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            config = {}
        else:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}

        old_lr = config.get("learning_rate", "Not Set")
        config["learning_rate"] = float(new_lr)

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)

        return f"Successfully updated learning rate from {old_lr} to {new_lr} in {config_path}. The training loop will pick this up on the next epoch."
    except Exception as e:
        return f"Failed to modify learning rate: {str(e)}"