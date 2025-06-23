def get_config(dataset_name):
    """Get configuration for different datasets"""
    
    base_config = {
        'ssl_lr': 0.001,
        'distill_lr': 0.001,
        'global_lr': 0.01,
        'local_lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'lambda_reg': 0.01,
        'num_models': 10,
        'online_iteration': 1000,
    }
    
    if dataset_name.lower() == 'mnist':
        config = {
            'num_classes': 10,
            'img_size': 28,
            'img_channels': 1,
            'feature_dim': 512,
            'ssl_feature_dim': 2048,
            'mean': (0.1307,),
            'std': (0.3081,),
        }
        
    elif dataset_name.lower() == 'fmnist':
        config = {
            'num_classes': 10,
            'img_size': 28,
            'img_channels': 1,
            'feature_dim': 512,
            'ssl_feature_dim': 2048,
            'mean': (0.2860,),
            'std': (0.3530,),
        }
        
    elif dataset_name.lower() == 'cifar10':
        config = {
            'num_classes': 10,
            'img_size': 32,
            'img_channels': 3,
            'feature_dim': 512,
            'ssl_feature_dim': 2048,
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2023, 0.1994, 0.2010),
        }
        
    elif dataset_name.lower() == 'svhn':
        config = {
            'num_classes': 10,
            'img_size': 32,
            'img_channels': 3,
            'feature_dim': 512,
            'ssl_feature_dim': 2048,
            'mean': (0.5, 0.5, 0.5),
            'std': (0.5, 0.5, 0.5),
        }
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Merge base config with dataset-specific config
    config.update(base_config)
    
    return config

# Dataset-specific transformations
def get_transforms(dataset_name, img_size, mean, std, augment=True):
    """Get data transformations for different datasets"""
    import torchvision.transforms as transforms
    
    if augment:
        if dataset_name.lower() in ['mnist', 'fmnist']:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:  # CIFAR-10, SVHN
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(img_size, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    return transform

# Augmentation strategies for different shift types
def get_shift_transform(shift_type, **kwargs):
    """Get transformations for different types of distribution shifts"""
    import torchvision.transforms as transforms
    
    if shift_type == 'rotation':
        angle = kwargs.get('angle', 0)
        return transforms.RandomRotation(degrees=angle)
    
    elif shift_type == 'noise':
        # Label noise is handled in data loading, not in transforms
        return transforms.Lambda(lambda x: x)
    
    else:
        return transforms.Lambda(lambda x: x)
