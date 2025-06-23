import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from collections import defaultdict
import random
from config import get_transforms, get_shift_transform

class RotationDataset(Dataset):
    """Dataset wrapper that applies rotation transformation"""
    def __init__(self, dataset, rotation_angle, dataset_name):
        self.dataset = dataset
        self.rotation_angle = rotation_angle
        self.dataset_name = dataset_name.lower()
        
        # Define rotation transform
        if self.dataset_name in ['mnist', 'fmnist']:
            self.rotation_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=(rotation_angle, rotation_angle)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,) if self.dataset_name == 'mnist' else (0.2860,), 
                    (0.3081,) if self.dataset_name == 'mnist' else (0.3530,)
                )
            ])
        else:
            self.rotation_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=(rotation_angle, rotation_angle)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465) if self.dataset_name == 'cifar10' else (0.5, 0.5, 0.5),
                    (0.2023, 0.1994, 0.2010) if self.dataset_name == 'cifar10' else (0.5, 0.5, 0.5)
                )
            ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Apply rotation transform
        if self.rotation_angle > 0:
            # Convert tensor back to PIL for rotation, then back to tensor
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            image = self.rotation_transform(image)
        
        return image, label


class NoisyDataset(Dataset):
    """Dataset wrapper that adds label noise"""
    def __init__(self, dataset, noise_rate, num_classes):
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self.noisy_labels = self._generate_noisy_labels()
        
    def _generate_noisy_labels(self):
        noisy_labels = []
        for i, (_, label) in enumerate(self.dataset):
            if random.random() < self.noise_rate:
                # Generate random label different from original
                noisy_label = random.randint(0, self.num_classes - 1)
                while noisy_label == label:
                    noisy_label = random.randint(0, self.num_classes - 1)
                noisy_labels.append(noisy_label)
            else:
                noisy_labels.append(label)
        return noisy_labels
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, self.noisy_labels[idx]

def dirichlet_partition(targets, num_clients, num_classes, alpha):
    """Partition data using Dirichlet distribution for label shift"""
    n_data = len(targets)
    client_dataidx_map = {}
    
    # Calculate proportions using Dirichlet distribution
    proportions = np.random.dirichlet(np.repeat(alpha, num_classes), num_clients)
    
    # Get indices for each class
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
    
    # Distribute indices to clients
    for client_id in range(num_clients):
        client_indices = []
        for class_id in range(num_classes):
            n_samples = int(proportions[client_id][class_id] * len(class_indices[class_id]))
            selected_indices = np.random.choice(
                class_indices[class_id], 
                size=min(n_samples, len(class_indices[class_id])), 
                replace=False
            )
            client_indices.extend(selected_indices)
            # Remove selected indices
            class_indices[class_id] = np.setdiff1d(class_indices[class_id], selected_indices)
        
        client_dataidx_map[client_id] = client_indices
    
    return client_dataidx_map

def create_rotation_transforms(num_clients, max_angle):
    """Create rotation transforms for feature shift"""
    transforms_list = []
    for i in range(num_clients):
        angle = (max_angle * i) / num_clients
        rotation_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(angle, angle)),
            transforms.ToTensor(),
        ])
        transforms_list.append(rotation_transform)
    return transforms_list

def get_dataset(dataset_name, num_clients, alpha=None, rotation_angle=0, noise_rate=0, batch_size=256):
    """
    Load dataset and create federated data loaders with different non-IID settings
    
    Args:
        dataset_name: Name of dataset ('mnist', 'cifar10', 'fmnist', 'svhn')
        num_clients: Number of federated clients
        alpha: Dirichlet parameter for label shift (smaller = more non-IID)
        rotation_angle: Maximum rotation angle for feature shift
        noise_rate: Label noise rate for concept shift
        batch_size: Batch size for data loaders
    
    Returns:
        train_loaders: List of training data loaders for each client
        test_loaders: List of test data loaders for each client  
        client_data_sizes: List of training data sizes for each client
    """
    
    # Define transforms based on dataset
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
    elif dataset_name.lower() == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        
    elif dataset_name.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
        
    elif dataset_name.lower() == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.SVHN('./data', split='train', download=True, transform=transform)
        test_dataset = datasets.SVHN('./data', split='test', download=True, transform=transform)
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Get number of classes
    if hasattr(train_dataset, 'classes'):
        num_classes = len(train_dataset.classes)
    else:
        num_classes = len(np.unique(train_dataset.targets if hasattr(train_dataset, 'targets') else train_dataset.labels))
    
    # Convert targets to numpy array for easier manipulation
    if hasattr(train_dataset, 'targets'):
        targets = np.array(train_dataset.targets)
    else:
        targets = np.array(train_dataset.labels)
    
    # Label shift: Dirichlet partition
    if alpha is not None:
        client_dataidx_map = dirichlet_partition(targets, num_clients, num_classes, alpha)
    else:
        # IID partition
        n_data = len(train_dataset)
        indices = np.random.permutation(n_data)
        split_size = n_data // num_clients
        client_dataidx_map = {}
        for i in range(num_clients):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < num_clients - 1 else n_data
            client_dataidx_map[i] = indices[start_idx:end_idx]
    
    # Create data loaders for each client
    train_loaders = []
    test_loaders = []
    client_data_sizes = []
    
    # Feature shift: rotation transforms
    rotation_transforms = None
    if rotation_angle > 0:
        rotation_transforms = create_rotation_transforms(num_clients, rotation_angle)
    
    for client_id in range(num_clients):
        # Get client's training data indices
        client_indices = client_dataidx_map[client_id]
        client_data_sizes.append(len(client_indices))
        
        # Create client's training dataset
        client_train_dataset = Subset(train_dataset, client_indices)
        
        # Apply feature shift (rotation) if specified
        if rotation_transforms is not None:
            # Create a wrapper dataset that applies rotation
            client_train_dataset = RotationDataset(
                client_train_dataset, 
                rotation_angle * client_id / num_clients,
                dataset_name
            )
        
        # Apply concept shift (label noise) if specified
        if noise_rate > 0:
            # Calculate client-specific noise rate
            client_noise_rate = (noise_rate * client_id) / num_clients
            client_train_dataset = NoisyDataset(client_train_dataset, client_noise_rate, num_classes)
        
        # Create data loader
        train_loader = DataLoader(
            client_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        train_loaders.append(train_loader)
        
        # Create test data loader (shared test set for all clients)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders, client_data_sizes

def select_clients_for_posfed_k(train_loaders, k):
    """
    Select K representative clients using clustering on label distributions
    
    Args:
        train_loaders: List of training data loaders
        k: Number of clients to select
    
    Returns:
        selected_indices: Indices of selected clients
    """
    # Simple implementation without sklearn dependency
    
    # Calculate label distribution for each client
    client_distributions = []
    num_classes = 10  # Assuming 10 classes for most datasets
    
    for loader in train_loaders:
        label_counts = np.zeros(num_classes)
        total_samples = 0
        
        for _, labels in loader:
            for label in labels:
                label_counts[label.item()] += 1
                total_samples += 1
        
        # Normalize to get distribution
        if total_samples > 0:
            distribution = label_counts / total_samples
        else:
            distribution = np.zeros(num_classes)
        client_distributions.append(distribution)
    
    client_distributions = np.array(client_distributions)
    
    # Simple clustering: select clients with most diverse distributions
    if k >= len(train_loaders):
        return list(range(len(train_loaders)))
    
    # Use a simple heuristic: select clients that maximize coverage of classes
    selected_indices = []
    remaining_clients = list(range(len(train_loaders)))
    
    # First, select client with most uniform distribution
    entropies = []
    for dist in client_distributions:
        # Calculate entropy (higher = more uniform)
        entropy = -np.sum(dist * np.log(dist + 1e-12))
        entropies.append(entropy)
    
    # Select client with highest entropy first
    first_client = np.argmax(entropies)
    selected_indices.append(first_client)
    remaining_clients.remove(first_client)
    
    # For remaining selections, pick clients that are most different from already selected
    for _ in range(k - 1):
        if not remaining_clients:
            break
            
        max_min_distance = -1
        best_client = None
        
        for candidate in remaining_clients:
            # Calculate minimum distance to any selected client
            min_distance = float('inf')
            for selected in selected_indices:
                # Use L2 distance between distributions
                distance = np.linalg.norm(client_distributions[candidate] - client_distributions[selected])
                min_distance = min(min_distance, distance)
            
            # Select candidate with maximum minimum distance (most diverse)
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_client = candidate
        
        if best_client is not None:
            selected_indices.append(best_client)
            remaining_clients.remove(best_client)
    
    return selected_indices
