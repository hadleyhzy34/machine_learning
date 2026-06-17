"""
Data loading utilities for CIFAR-10.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 2,
    device: str = "cpu",
    data_root: str = "./data",
):
    """
    Create train and test dataloaders for CIFAR-10.

    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        device: Device being used (MPS requires num_workers=0)
        data_root: Root directory for data storage

    Returns:
        train_loader, test_loader, class_names
    """
    # MPS doesn't support multiprocessing
    if device == "mps" and num_workers > 0:
        num_workers = 0

    # CIFAR-10 normalization values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)

    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Test transforms (only normalization)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    # Pin memory only for CUDA
    pin_memory = device == "cuda"

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, train_dataset.classes
