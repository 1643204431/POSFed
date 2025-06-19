# POSFed: Personalized One-Shot Federated Learning

This is the implementation of POSFed framework from the paper "POSFed: Tackling Non-IID Challenges in One-Shot Federated Learning via Personalization".

## Requirements

```
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
```

## Quick Start

### Basic Usage
```bash
python main.py --dataset cifar10 --num_clients 100 --alpha 0.1
```

### Dataset Options
- mnist
- cifar10  
- fmnist
- svhn

### Non-IID Settings
- `--alpha`: Dirichlet parameter for label shift (0.1, 0.3, 0.5)
- `--rotation_angle`: Maximum rotation angle for feature shift (15, 30, 45)
- `--noise_rate`: Label noise rate for concept shift (0.01, 0.03, 0.05)

### POSFed-K (Communication Reduction)
```bash
python main.py --dataset cifar10 --posfed_k 10 --num_clients 100
```

## File Structure

```
├── main.py                 # Main entry point
├── algorithms/
│   ├── posfed.py          # POSFed algorithm implementation
│   └── krr_st.py          # KRR-ST dataset distillation
├── data/
│   └── data_loader.py     # Data loading and non-IID partition
├── models/
│   └── networks.py        # Network architectures
├── utils/
│   └── utils.py           # Utility functions
└── config.py              # Configuration settings
```

## Key Parameters

- `--num_clients`: Number of federated clients (default: 100)
- `--synthetic_per_client`: Synthetic samples per client (default: 100)
- `--global_epochs`: Global feature extractor training epochs (default: 1000)
- `--local_epochs`: Local personalization epochs (default: 100)
- `--learning_rate`: Learning rate (default: 0.01)

## Results

The implementation reproduces the results from the paper across different non-IID scenarios on MNIST, Fashion-MNIST, CIFAR-10, and SVHN datasets.
