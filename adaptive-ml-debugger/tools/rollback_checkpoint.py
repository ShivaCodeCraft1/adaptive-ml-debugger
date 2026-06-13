import os
import shutil
from langchain_core.tools import tool

@tool
def rollback_checkpoint(checkpoint_dir: str = "./checkpoints", target_file: str = "best_checkpoint.pt") -> str:
    """
    Rolls back the current model to a stable checkpoint (e.g., 'best_checkpoint.pt').
    Useful when the model has irrecoverably diverged or heavily overfitted.
    
    Args:
        checkpoint_dir (str): Directory where checkpoints are stored.
        target_file (str): The stable checkpoint file to rollback to.
        
    Returns:
        str: Status of the rollback operation.
    """
    try:
        stable_path = os.path.join(checkpoint_dir, target_file)
        latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        rollback_flag_path = os.path.join(checkpoint_dir, "rollback_flag.txt")

        if not os.path.exists(stable_path):
            return f"Cannot rollback: '{target_file}' not found in {checkpoint_dir}."

        # Simulate rollback by copying the stable checkpoint over the latest one
        shutil.copy2(stable_path, latest_path)
        
        # Drop a flag file so the training loop knows to reload state from disk
        with open(rollback_flag_path, "w") as f:
            f.write(f"Rolled back to {target_file}")

        return f"Successfully rolled back to {target_file}. The training loop has been flagged to reload this checkpoint."
    except Exception as e:
        return f"Failed to rollback checkpoint: {str(e)}"