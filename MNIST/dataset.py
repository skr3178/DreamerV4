"""
MNIST Dataset for Shortcut Model Testing

Creates sequences of MNIST images for testing the shortcut forcing objective.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class MNISTSequenceDataset(Dataset):
    """
    MNIST dataset that creates sequences of images for testing shortcut forcing.
    
    Each sample is a sequence of MNIST images (frames) that can be used to test
    the dynamics model's ability to predict future frames.
    """
    
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        sequence_length: int = 8,
        image_size: tuple = (28, 28),
        download: bool = True,
    ):
        """
        Args:
            root: Root directory for MNIST data
            train: Whether to use training or test set
            sequence_length: Number of frames per sequence
            image_size: Target image size (height, width)
            download: Whether to download MNIST if not present
        """
        self.sequence_length = sequence_length
        self.image_size = image_size
        
        # Load MNIST dataset
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            # Normalize to [-1, 1] range (common for diffusion models)
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
        
        # Create sequences by grouping consecutive samples
        # For simplicity, we'll create sequences by taking consecutive images
        # In a real scenario, you might want to create sequences based on digit transitions
        self.num_sequences = len(self.mnist) // sequence_length
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """
        Returns a sequence of MNIST images.
        
        Returns:
            frames: (sequence_length, 1, H, W) tensor of images
            labels: (sequence_length,) tensor of digit labels (for reference)
        """
        start_idx = idx * self.sequence_length
        end_idx = start_idx + self.sequence_length
        
        frames = []
        labels = []
        
        for i in range(start_idx, end_idx):
            if i < len(self.mnist):
                img, label = self.mnist[i]
                frames.append(img)
                labels.append(label)
            else:
                # Pad with last image if needed
                frames.append(frames[-1] if frames else torch.zeros(1, *self.image_size))
                labels.append(labels[-1] if labels else 0)
        
        frames = torch.stack(frames)  # (T, C, H, W)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            "frames": frames,
            "labels": labels,
        }


def create_mnist_dataloader(
    batch_size: int = 32,
    sequence_length: int = 8,
    image_size: tuple = (28, 28),
    train: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for MNIST sequences.
    
    Args:
        batch_size: Batch size
        sequence_length: Number of frames per sequence
        image_size: Target image size
        train: Whether to use training or test set
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        DataLoader for MNIST sequences
    """
    dataset = MNISTSequenceDataset(
        train=train,
        sequence_length=sequence_length,
        image_size=image_size,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )
