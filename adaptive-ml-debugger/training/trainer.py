import os
import torch
import torch.nn as nn
from typing import Dict, Any

class Trainer:
    """
    Manages the training loop, validation loop, gradient tracking, 
    and checkpoint saving for the PyTorch model.
    """
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module, 
        device: torch.device, 
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _compute_grad_norm(self) -> float:
        """
        Computes the L2 norm of the gradients across all model parameters.
        Used by the agent to detect exploding or vanishing gradients.
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        correct = 0
        total = 0

        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Track gradient norm before optimizer step
            grad_norm = self._compute_grad_norm()
            total_grad_norm += grad_norm
            
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        num_batches = len(dataloader)
        metrics = {
            "train_loss": total_loss / num_batches,
            "train_acc": correct / total,
            "avg_grad_norm": total_grad_norm / num_batches
        }
        return metrics

    def validate_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Runs one epoch of validation.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                total_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        metrics = {
            "val_loss": total_loss / len(dataloader),
            "val_acc": correct / total
        }
        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], filename: str = "checkpoint.pt") -> str:
        """
        Saves the model and optimizer state.
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(state, filepath)
        return filepath