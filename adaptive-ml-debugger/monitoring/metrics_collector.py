from typing import Dict, Any, Optional
from monitoring.mlflow_logger import MLFlowLogger
from monitoring.prometheus_exporter import prometheus_exporter

class MetricsCollector:
    """
    An orchestrator class that unifies logging across MLflow, Prometheus, 
    and potentially other systems (like Weights & Biases) in one interface.
    """
    def __init__(self, experiment_name: str = "adaptive-ml-debugger"):
        self.mlflow_logger = MLFlowLogger(experiment_name=experiment_name)
        self.prometheus = prometheus_exporter

    def start_run(self, config: Dict[str, Any], run_name: Optional[str] = None) -> None:
        """
        Initializes the tracking run and logs initial configurations.
        """
        self.mlflow_logger.start_run(run_name=run_name)
        self.mlflow_logger.log_params(config)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Logs epoch-level metrics to both MLflow and Prometheus.
        """
        # 1. Log to MLflow (Historical Tracking)
        self.mlflow_logger.log_metrics(metrics, step=epoch)
        
        # 2. Update Prometheus (Real-time Scraping)
        self.prometheus.update_metrics(metrics, epoch)

    def log_checkpoint(self, filepath: str) -> None:
        """
        Logs a model checkpoint to MLflow.
        """
        self.mlflow_logger.log_artifact(filepath, artifact_path="checkpoints")

    def record_anomaly(self) -> None:
        """
        Records that an anomaly occurred in the real-time Prometheus metrics.
        """
        self.prometheus.increment_anomaly_counter()

    def end_run(self) -> None:
        """
        Safely closes all tracking runs.
        """
        self.mlflow_logger.end_run()