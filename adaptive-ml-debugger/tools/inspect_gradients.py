from langchain_core.tools import tool
import mlflow
from mlflow.tracking import MlflowClient

@tool
def inspect_gradients(run_id: str) -> str:
    """
    Inspects the gradient norms for the current training run to check for 
    vanishing or exploding gradients. Requires the MLflow run_id.
    
    Args:
        run_id (str): The ID of the active MLflow run.
        
    Returns:
        str: A report on the status of the gradients over the last few epochs.
    """
    try:
        client = MlflowClient()
        history = client.get_metric_history(run_id, "avg_grad_norm")
        
        if not history:
            return "No gradient history found for this run. Training might not have progressed enough."
        
        # Extract the last 5 gradient norms
        recent_grads = [m.value for m in history[-5:]]
        
        report = f"Recent gradient norms (last 5 epochs): {recent_grads}\n"
        
        if any(g > 10.0 for g in recent_grads):
            report += "Observation: Gradients appear to be exploding (>10.0)."
        elif all(g < 1e-4 for g in recent_grads):
            report += "Observation: Gradients appear to be vanishing (<1e-4)."
        else:
            report += "Observation: Gradients appear stable."
            
        return report
    except Exception as e:
        return f"Error inspecting gradients from MLflow: {str(e)}"