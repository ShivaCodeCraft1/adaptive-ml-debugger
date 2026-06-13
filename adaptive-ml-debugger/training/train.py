import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, Any

from training.model import create_model
from training.dataset import get_dataloaders
from training.trainer import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_training(config: Dict[str, Any]) -> None:
    """
    Main entry point for starting a training run. Hooks into the dataset, 
    model initialization, and the Trainer class.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training on device: {device}")

    # Prepare Data
    train_loader, val_loader = get_dataloaders(
        batch_size=config.get("batch_size", 64),
        data_dir=config.get("data_dir", "./data")
    )

    # Prepare Model
    model = create_model(config)
    logger.info(f"Initialized model with {sum(p.numel() for p in model.parameters())} parameters.")

    # Prepare Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))
    criterion = nn.CrossEntropyLoss()

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=config.get("checkpoint_dir", "./checkpoints")
    )

    epochs = config.get("epochs", 10)
    best_val_loss = float('inf')

    # Training Loop
    for epoch in range(1, epochs + 1):
        logger.info(f"--- Epoch {epoch}/{epochs} ---")
        
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate_epoch(val_loader)

        # Log metrics
        logger.info(
            f"Train | Loss: {train_metrics['train_loss']:.4f} | "
            f"Acc: {train_metrics['train_acc']:.4f} | "
            f"Grad Norm: {train_metrics['avg_grad_norm']:.4f}"
        )
        logger.info(
            f"Val   | Loss: {val_metrics['val_loss']:.4f} | "
            f"Acc: {val_metrics['val_acc']:.4f}"
        )

        # Merge metrics for checkpoint state
        all_metrics = {**train_metrics, **val_metrics}
        
        # Always save the latest state
        trainer.save_checkpoint(epoch, all_metrics, filename="latest_checkpoint.pt")

        # Save best model based on validation loss
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            trainer.save_checkpoint(epoch, all_metrics, filename="best_checkpoint.pt")
            logger.info(f"New best validation loss: {best_val_loss:.4f}. Checkpoint saved.")

if __name__ == "__main__":
    # Example local run configuration
    default_config = {
        "input_size": 784,
        "hidden_sizes": [256, 128],
        "output_size": 10,
        "dropout_rate": 0.1,
        "activation": "relu",
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 10,
        "data_dir": "./data",
        "checkpoint_dir": "./checkpoints"
    }
    
    run_training(default_config)