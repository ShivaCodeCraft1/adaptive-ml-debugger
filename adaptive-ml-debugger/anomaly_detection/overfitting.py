from typing import Dict, Any, List

def detect_overfitting(
    train_losses: List[float], 
    val_losses: List[float], 
    patience: int = 3
) -> Dict[str, Any]:
    """
    Detects overfitting by observing if validation loss increases consecutively 
    while training loss decreases for a specified number of epochs.
    
    Args:
        train_losses (List[float]): History of training losses.
        val_losses (List[float]): History of validation losses.
        patience (int): Number of consecutive epochs validation loss must increase.
        
    Returns:
        Dict[str, Any]: A dictionary containing a boolean flag 'detected' and a 'message'.
    """
    if len(train_losses) < patience + 1 or len(val_losses) < patience + 1:
        return {"detected": False, "message": "Not enough data to detect overfitting."}
        
    # Isolate the most recent sequence
    recent_val = val_losses[-(patience + 1):]
    recent_train = train_losses[-(patience + 1):]
    
    # Check if validation loss has been strictly increasing
    val_increasing = all(recent_val[i] < recent_val[i+1] for i in range(patience))
    
    # Check if training loss has been strictly decreasing
    train_decreasing = all(recent_train[i] > recent_train[i+1] for i in range(patience))
    
    if val_increasing and train_decreasing:
        return {
            "detected": True, 
            "message": f"Overfitting detected: validation loss increased while training loss decreased for {patience} consecutive epochs."
        }
            
    return {"detected": False, "message": "No strict overfitting pattern detected."}