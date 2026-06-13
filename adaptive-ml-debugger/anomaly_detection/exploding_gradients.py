import math
from typing import Dict, Any, List

def detect_exploding_gradients(grad_norms: List[float], threshold: float = 10.0) -> Dict[str, Any]:
    """
    Detects if the gradients are exploding by checking if the latest gradient norm
    exceeds a defined threshold, or if it has reached NaN/Infinity.
    
    Args:
        grad_norms (List[float]): A history of gradient norms.
        threshold (float): The threshold above which a gradient is considered exploding.
        
    Returns:
        Dict[str, Any]: A dictionary containing a boolean flag 'detected' and a 'message'.
    """
    if not grad_norms:
        return {"detected": False, "message": "No gradient data available."}
        
    latest_grad = grad_norms[-1]
    
    if math.isnan(latest_grad) or math.isinf(latest_grad):
        return {
            "detected": True, 
            "message": f"Exploding gradient detected: value is NaN or Infinity."
        }
        
    if latest_grad > threshold:
        return {
            "detected": True, 
            "message": f"Exploding gradient detected: norm {latest_grad:.4f} exceeds threshold of {threshold}."
        }
        
    return {"detected": False, "message": "Gradients are stable."}