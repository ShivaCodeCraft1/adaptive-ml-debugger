from prometheus_client import Gauge, Counter, REGISTRY
from typing import Dict

class PrometheusExporter:
    """
    Maintains Prometheus metrics for the training job. These metrics 
    can be scraped by a Prometheus server via the FastAPI /metrics endpoint.
    """
    def __init__(self):
        # We use Gauges because these values go up and down during training
        self.train_loss = Gauge('model_train_loss', 'Training loss per epoch')
        self.val_loss = Gauge('model_val_loss', 'Validation loss per epoch')
        self.train_acc = Gauge('model_train_accuracy', 'Training accuracy per epoch')
        self.val_acc = Gauge('model_val_accuracy', 'Validation accuracy per epoch')
        self.grad_norm = Gauge('model_gradient_norm', 'Average gradient L2 norm')
        self.epoch_gauge = Gauge('model_current_epoch', 'Current training epoch')
        
        # A counter for anomalies detected by the LangGraph agent
        self.anomalies_detected = Counter('model_anomalies_detected_total', 'Total number of training anomalies detected')

    def update_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        Updates the Prometheus Gauges with the latest training metrics.
        """
        if 'train_loss' in metrics:
            self.train_loss.set(metrics['train_loss'])
        
        if 'val_loss' in metrics:
            self.val_loss.set(metrics['val_loss'])
            
        if 'train_acc' in metrics:
            self.train_acc.set(metrics['train_acc'])
            
        if 'val_acc' in metrics:
            self.val_acc.set(metrics['val_acc'])
            
        if 'avg_grad_norm' in metrics:
            self.grad_norm.set(metrics['avg_grad_norm'])
            
        self.epoch_gauge.set(epoch)

    def increment_anomaly_counter(self) -> None:
        """
        Increments the counter when the agent detects an anomaly.
        """
        self.anomalies_detected.inc()

# Singleton instance to be imported and used by FastAPI and Training loops
prometheus_exporter = PrometheusExporter()