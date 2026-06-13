from typing import Dict, Any, List

def detect_vanishing_gradients(
    grad_norms: List[float], 
    threshold: float = 1e-4, 
    window: int = 3
) -> Dict[str, Any]:
    """
    Detects vanishing gradients by checking if the gradient norm has been 
    consistently below a very small threshold for a specified number of epochs/steps.
    
    Args:
        grad_norms (List[float]): A history of gradient norms.
        threshold (float): The threshold below which a gradient is considered vanishing.
        window (int): The number of consecutive epochs the condition must hold.
        
    Returns:
        Dict[str, Any]: A dictionary containing a boolean flag 'detected' and a 'message'.
    """
    if len(grad_norms) < window:
        return {"detected": False, "message": "Not enough gradient data to detect vanishing gradients."}
        
    recent_grads = grad_norms[-window:]
    
    if all(g < threshold for g in recent_grads):
        return {
            "detected": True, 
            "message": f"Vanishing gradients detected: last {window} norms strictly below {threshold}."
        }
        
    return {"detected": False, "message": "Gradients are active."}