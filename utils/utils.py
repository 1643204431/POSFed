import torch
import numpy as np
import random
import os
import json
from datetime import datetime

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        # Return single value for k=1 case
        if len(res) == 1:
            return res[0]
        return res

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_results(results, filename):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable_results[key] = value.item()
        else:
            serializable_results[key] = value
    
    # Add timestamp
    serializable_results['timestamp'] = datetime.now().isoformat()
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filename}")

def load_results(filename):
    """Load results from JSON file"""
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

def calculate_statistics(values):
    """Calculate mean, std, min, max of a list of values"""
    values = np.array(values)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values))
    }

def print_results_table(results_dict, title="Results"):
    """Print results in a formatted table"""
    print(f"\n{title}")
    print("=" * 60)
    
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            if isinstance(value, (int, float)):
                print(f"{key:30s}: {value:8.4f}")
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                stats = calculate_statistics(value)
                print(f"{key:30s}: {stats['mean']:8.4f} ± {stats['std']:8.4f}")
            else:
                print(f"{key:30s}: {value}")
    
    print("=" * 60)

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(model, model_name="Model"):
    """Print model information"""
    num_params = count_parameters(model)
    print(f"{model_name} - Parameters: {num_params:,}")

class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def create_exp_name(config):
    """Create experiment name based on configuration"""
    dataset = config.get('dataset', 'unknown')
    alpha = config.get('alpha', 'none')
    num_clients = config.get('num_clients', 'unknown')
    posfed_k = config.get('posfed_k', None)
    
    exp_name = f"posfed_{dataset}_alpha{alpha}_clients{num_clients}"
    
    if posfed_k:
        exp_name += f"_k{posfed_k}"
    
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name += f"_{timestamp}"
    
    return exp_name

def log_config(config, log_file=None):
    """Log configuration to file"""
    config_str = "Configuration:\n"
    config_str += "=" * 40 + "\n"
    
    for key, value in sorted(config.items()):
        config_str += f"{key:25s}: {value}\n"
    
    config_str += "=" * 40 + "\n"
    
    if log_file:
        with open(log_file, 'w') as f:
            f.write(config_str)
    
    print(config_str)

def compare_methods(results_list, method_names, metric='avg_test_acc'):
    """Compare multiple methods and print results"""
    print(f"\nMethod Comparison - {metric}")
    print("=" * 60)
    
    for i, (results, name) in enumerate(zip(results_list, method_names)):
        value = results.get(metric, 0)
        std = results.get(f'std_test_acc', 0) if 'avg' in metric else 0
        
        if std > 0:
            print(f"{name:20s}: {value:8.4f} ± {std:8.4f}")
        else:
            print(f"{name:20s}: {value:8.4f}")
    
    print("=" * 60)

def analyze_non_iid_severity(train_loaders, num_classes=10):
    """Analyze the severity of non-IID distribution"""
    client_distributions = []
    
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
    
    # Calculate statistics
    stats = {
        'mean_entropy': np.mean([entropy(dist) for dist in client_distributions]),
        'std_entropy': np.std([entropy(dist) for dist in client_distributions]),
        'mean_classes_per_client': np.mean([np.sum(dist > 0) for dist in client_distributions]),
        'kl_divergence_from_uniform': np.mean([kl_divergence(dist, np.ones(num_classes)/num_classes) 
                                             for dist in client_distributions])
    }
    
    return stats

def entropy(p):
    """Calculate entropy of a probability distribution"""
    p = p + 1e-12  # Add small constant to avoid log(0)
    return -np.sum(p * np.log(p))

def kl_divergence(p, q):
    """Calculate KL divergence between two distributions"""
    p = p + 1e-12
    q = q + 1e-12
    return np.sum(p * np.log(p / q))

def create_summary_report(results, config, save_path=None):
    """Create a comprehensive summary report"""
    report = []
    report.append("POSFed Experiment Summary")
    report.append("=" * 50)
    report.append("")
    
    # Configuration
    report.append("Configuration:")
    report.append("-" * 20)
    for key, value in sorted(config.items()):
        report.append(f"{key}: {value}")
    report.append("")
    
    # Results
    report.append("Results:")
    report.append("-" * 20)
    report.append(f"Average Test Accuracy: {results['avg_test_acc']:.4f}")
    report.append(f"Standard Deviation: {results['std_test_acc']:.4f}")
    report.append("")
    
    # Client-wise results
    if 'client_accuracies' in results:
        report.append("Client-wise Accuracies:")
        report.append("-" * 20)
        for i, acc in enumerate(results['client_accuracies']):
            report.append(f"Client {i:3d}: {acc:6.2f}%")
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Summary report saved to {save_path}")
    
    return report_text

# Learning rate schedulers
def get_scheduler(optimizer, scheduler_type='cosine', **kwargs):
    """Get learning rate scheduler"""
    if scheduler_type == 'cosine':
        T_max = kwargs.get('T_max', 100)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'exponential':
        gamma = kwargs.get('gamma', 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        return None

# Memory management
def clear_cache():
    """Clear GPU cache if available"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB"
    else:
        return "CPU mode - no GPU memory tracking"
