import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple

def get_dataloaders(
    batch_size: int = 64, 
    data_dir: str = "./data", 
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Downloads and prepares the MNIST dataset, returning train and validation dataloaders.
    
    Args:
        batch_size (int): The batch size for training and validation.
        data_dir (str): Directory to store the downloaded dataset.
        num_workers (int): Number of subprocesses to use for data loading.
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train dataloader and Validation dataloader.
    """
    # Standard MNIST transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    val_dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader