from typing import Dict, Any, List

def detect_divergence(
    train_losses: List[float], 
    val_losses: List[float], 
    ratio_threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Detects severe train/validation divergence. This indicates catastrophic 
    instability rather than gradual overfitting (e.g., val loss spiking massively).
    
    Args:
        train_losses (List[float]): History of training losses.
        val_losses (List[float]): History of validation losses.
        ratio_threshold (float): How many times larger the val loss can be compared to train loss.
        
    Returns:
        Dict[str, Any]: A dictionary containing a boolean flag 'detected' and a 'message'.
    """
    if not train_losses or not val_losses:
        return {"detected": False, "message": "Insufficient loss data."}
        
    latest_train = train_losses[-1]
    latest_val = val_losses[-1]
    
    # Prevent division by zero
    safe_train = max(latest_train, 1e-8)
        
    ratio = latest_val / safe_train
    
    if ratio > ratio_threshold:
        return {
            "detected": True, 
            "message": f"Divergence detected: validation loss is {ratio:.2f}x the training loss."
        }
             
    return {"detected": False, "message": "Train and validation losses are aligned."}