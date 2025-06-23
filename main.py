import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from algorithms.posfed import POSFedServer
from data.data_loader import get_dataset
from config import get_config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='POSFed: Personalized One-Shot Federated Learning')
    
    # Dataset and model
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['mnist', 'cifar10', 'fmnist', 'svhn'])
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'])
    
    # Federated learning settings
    parser.add_argument('--num_clients', type=int, default=100)
    parser.add_argument('--synthetic_per_client', type=int, default=100)
    parser.add_argument('--posfed_k', type=int, default=None, 
                       help='Number of selected clients for POSFed-K variant')
    
    # Non-IID settings
    parser.add_argument('--alpha', type=float, default=0.1, 
                       help='Dirichlet parameter for label shift')
    parser.add_argument('--rotation_angle', type=float, default=0, 
                       help='Maximum rotation angle for feature shift')
    parser.add_argument('--noise_rate', type=float, default=0, 
                       help='Label noise rate for concept shift')
    
    # Training parameters
    parser.add_argument('--ssl_epochs', type=int, default=100, 
                       help='Self-supervised pretraining epochs')
    parser.add_argument('--distill_iterations', type=int, default=1000, 
                       help='Dataset distillation iterations')
    parser.add_argument('--global_epochs', type=int, default=1000, 
                       help='Global feature extractor training epochs')
    parser.add_argument('--local_epochs', type=int, default=100, 
                       help='Local personalization epochs')
    
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    # KRR-ST parameters
    parser.add_argument('--num_models', type=int, default=10, 
                       help='Number of models in model pool')
    parser.add_argument('--online_iteration', type=int, default=1000, 
                       help='Online training iterations for model pool')
    parser.add_argument('--lambda_reg', type=float, default=0.01, 
                       help='Regularization parameter for KRR')
    
    # Experiment settings
    parser.add_argument('--save_results', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Update config with args
    config = get_config(args.dataset)
    
    # Update config with command line arguments
    for key, value in vars(args).items():
        config[key] = value
    
    if args.verbose:
        print("="*50)
        print("POSFed: Personalized One-Shot Federated Learning")
        print("="*50)
        print(f"Dataset: {args.dataset}")
        print(f"Clients: {args.num_clients}")
        print(f"Non-IID settings: α={args.alpha}, rotation={args.rotation_angle}°, noise={args.noise_rate}")
        if args.posfed_k:
            print(f"POSFed-K with {args.posfed_k} selected clients")
        print("="*50)
    
    # Load data and create federated datasets
    train_loaders, test_loaders, client_data_sizes = get_dataset(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        alpha=args.alpha,
        rotation_angle=args.rotation_angle,
        noise_rate=args.noise_rate,
        batch_size=args.batch_size
    )
    
    # Create POSFed server
    server = POSFedServer(
        config=config,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        client_data_sizes=client_data_sizes,
        device=args.device
    )
    
    # Run POSFed algorithm
    results = server.run()
    
    if args.verbose:
        print("\n" + "="*50)
        print("Results:")
        print(f"Average test accuracy: {results['avg_test_acc']:.4f}")
        print(f"Standard deviation: {results['std_test_acc']:.4f}")
        print("="*50)
    
    # Save results
    if args.save_results:
        import json
        import os
        
        os.makedirs('results', exist_ok=True)
        result_file = f"results/posfed_{args.dataset}_alpha{args.alpha}_clients{args.num_clients}.json"
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if args.verbose:
            print(f"Results saved to {result_file}")

if __name__ == '__main__':
    main()
