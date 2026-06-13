import os
import mlflow
from typing import Dict, Any, Optional

class MLFlowLogger:
    """
    Handles logging of parameters, metrics, and models to an MLflow tracking server.
    """
    def __init__(self, experiment_name: str = "adaptive-ml-debugger"):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.active_run = None

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        Starts a new MLflow run.
        """
        self.active_run = mlflow.start_run(run_name=run_name)
        return self.active_run

    def end_run(self) -> None:
        """
        Ends the current MLflow run.
        """
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Logs hyperparameter configurations.
        """
        if not self.active_run:
            self.start_run()
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Logs performance metrics (loss, accuracy, grad norm) for a specific step/epoch.
        """
        if not self.active_run:
            self.start_run()
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Logs local files (e.g., checkpoints, config files) as artifacts.
        """
        if not self.active_run:
            self.start_run()
        mlflow.log_artifact(local_path, artifact_path)