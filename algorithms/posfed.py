import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from algorithms.krr_st import KRRSTDistiller
from models.networks import get_model, ClassificationHead
from data.data_loader import select_clients_for_posfed_k
from utils.utils import AverageMeter, accuracy

class POSFedServer:
    """POSFed Server Implementation"""
    
    def __init__(self, config, train_loaders, test_loaders, client_data_sizes, device):
        self.config = config
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.client_data_sizes = client_data_sizes
        self.device = device
        
        self.num_clients = len(train_loaders)
        self.num_classes = config['num_classes']
        self.feature_dim = config['feature_dim']
        
        # POSFed-K setting
        self.posfed_k = config.get('posfed_k', None)
        self.selected_clients = None
        
        # Training parameters
        self.synthetic_per_client = config.get('synthetic_per_client', 100)
        self.global_epochs = config.get('global_epochs', 1000)
        self.local_epochs = config.get('local_epochs', 100)
        self.global_lr = config.get('global_lr', 0.01)
        self.local_lr = config.get('local_lr', 0.01)
        
        # Initialize models
        self.global_feature_extractor = None
        self.client_classifiers = []
        
        print(f"POSFed Server initialized with {self.num_clients} clients")
        if self.posfed_k:
            print(f"POSFed-K enabled with k={self.posfed_k} selected clients")
    
    def run(self):
        """Run the complete POSFed algorithm"""
        
        print("\n" + "="*60)
        print("Starting POSFed Algorithm")
        print("="*60)
        
        # Stage 1: Self-Supervised Dataset Distillation
        print("\nStage 1: Self-Supervised Dataset Distillation")
        synthetic_datasets = self.stage1_dataset_distillation()
        
        # Stage 2: Global Feature Extractor Learning  
        print("\nStage 2: Global Feature Extractor Learning")
        self.global_feature_extractor = self.stage2_global_training(synthetic_datasets)
        
        # Stage 3: Client-Specific Personalization
        print("\nStage 3: Client-Specific Personalization")
        results = self.stage3_personalization()
        
        return results
    
    def stage1_dataset_distillation(self):
        """Stage 1: Each client performs self-supervised dataset distillation"""
        
        # Select clients for POSFed-K
        if self.posfed_k and self.posfed_k < self.num_clients:
            print(f"Selecting {self.posfed_k} representative clients...")
            self.selected_clients = select_clients_for_posfed_k(self.train_loaders, self.posfed_k)
            print(f"Selected clients: {self.selected_clients}")
        else:
            self.selected_clients = list(range(self.num_clients))
        
        # Initialize KRR-ST distiller
        distiller = KRRSTDistiller(self.config, self.device)
        
        synthetic_datasets = {}
        
        print(f"Running dataset distillation for {len(self.selected_clients)} clients...")
        
        for i, client_id in enumerate(self.selected_clients):
            print(f"\nClient {client_id} ({i+1}/{len(self.selected_clients)})")
            
            # Get client's training data
            train_loader = self.train_loaders[client_id]
            
            # Perform KRR-ST distillation
            x_syn, y_syn = distiller.run(train_loader, self.synthetic_per_client)
            
            synthetic_datasets[client_id] = {
                'x_syn': x_syn,
                'y_syn': y_syn
            }
            
            print(f"Client {client_id}: Generated {x_syn.shape[0]} synthetic samples")
        
        return synthetic_datasets
    
    def stage2_global_training(self, synthetic_datasets):
        """Stage 2: Train global feature extractor on aggregated synthetic data"""
        
        print("Aggregating synthetic datasets...")
        
        # Aggregate all synthetic datasets
        all_x_syn = []
        all_y_syn = []
        
        for client_id, data in synthetic_datasets.items():
            all_x_syn.append(data['x_syn'])
            all_y_syn.append(data['y_syn'])
        
        # Concatenate all synthetic data
        x_syn_agg = torch.cat(all_x_syn, dim=0)
        y_syn_agg = torch.cat(all_y_syn, dim=0)
        
        print(f"Aggregated synthetic dataset size: {x_syn_agg.shape[0]} samples")
        
        # Create global feature extractor
        global_model = get_model('cnn', self.config).to(self.device)
        
        # Optimizer for global training
        optimizer = torch.optim.SGD(
            global_model.parameters(), 
            lr=self.global_lr, 
            momentum=0.9, 
            weight_decay=5e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.global_epochs
        )
        
        # Training loop
        global_model.train()
        
        print("Training global feature extractor...")
        
        for epoch in trange(self.global_epochs, desc="Global Training"):
            
            # Forward pass
            pred = global_model(x_syn_agg)
            
            # MSE loss for regression to self-supervised targets
            loss = F.mse_loss(pred, y_syn_agg)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Log progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        print("Global feature extractor training completed!")
        
        return global_model
    
    def stage3_personalization(self):
        """Stage 3: Train personalized classification heads for each client"""
        
        print("Training personalized classification heads...")
        
        # Freeze global feature extractor
        for param in self.global_feature_extractor.parameters():
            param.requires_grad = False
        
        self.global_feature_extractor.eval()
        
        client_accuracies = []
        self.client_classifiers = []
        
        for client_id in range(self.num_clients):
            print(f"\nTraining client {client_id} classifier...")
            
            # Create personalized classification head
            classifier = ClassificationHead(self.feature_dim, self.num_classes).to(self.device)
            
            # Optimizer for local training
            optimizer = torch.optim.SGD(
                classifier.parameters(),
                lr=self.local_lr,
                momentum=0.9,
                weight_decay=0.0
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.local_epochs
            )
            
            # Training data for this client
            train_loader = self.train_loaders[client_id]
            test_loader = self.test_loaders[client_id]
            
            # Training loop
            classifier.train()
            
            for epoch in range(self.local_epochs):
                train_loss = AverageMeter()
                train_acc = AverageMeter()
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Extract features using frozen global feature extractor
                    with torch.no_grad():
                        features = self.global_feature_extractor(data)
                    
                    # Forward pass through classification head
                    output = classifier(features)
                    loss = F.cross_entropy(output, target)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    acc = accuracy(output, target)[0]
                    train_loss.update(loss.item(), data.size(0))
                    train_acc.update(acc.item(), data.size(0))
                
                scheduler.step()
                
                # Log progress
                if epoch % 20 == 0:
                    print(f"Client {client_id}, Epoch {epoch}, "
                          f"Loss: {train_loss.avg:.4f}, Acc: {train_acc.avg:.2f}%")
            
            # Evaluate on test set
            test_acc = self.evaluate_client(client_id, classifier, test_loader)
            client_accuracies.append(test_acc)
            self.client_classifiers.append(classifier)
            
            print(f"Client {client_id} test accuracy: {test_acc:.2f}%")
        
        # Compute overall results
        avg_test_acc = np.mean(client_accuracies)
        std_test_acc = np.std(client_accuracies)
        
        results = {
            'client_accuracies': client_accuracies,
            'avg_test_acc': avg_test_acc,
            'std_test_acc': std_test_acc,
            'selected_clients': self.selected_clients,
            'config': self.config
        }
        
        print(f"\nOverall Results:")
        print(f"Average test accuracy: {avg_test_acc:.2f}%")
        print(f"Standard deviation: {std_test_acc:.2f}%")
        
        return results
    
    def evaluate_client(self, client_id, classifier, test_loader):
        """Evaluate a client's personalized model"""
        
        self.global_feature_extractor.eval()
        classifier.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Extract features
                features = self.global_feature_extractor(data)
                
                # Classify
                output = classifier(features)
                
                # Count correct predictions
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / total
        return accuracy
    
    def save_model(self, save_path):
        """Save the trained models"""
        import os
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save global feature extractor
        torch.save(
            self.global_feature_extractor.state_dict(),
            os.path.join(save_path, 'global_feature_extractor.pth')
        )
        
        # Save client classifiers
        for i, classifier in enumerate(self.client_classifiers):
            torch.save(
                classifier.state_dict(),
                os.path.join(save_path, f'client_{i}_classifier.pth')
            )
        
        print(f"Models saved to {save_path}")
    
    def load_model(self, save_path):
        """Load pre-trained models"""
        import os
        
        # Load global feature extractor
        global_path = os.path.join(save_path, 'global_feature_extractor.pth')
        if os.path.exists(global_path):
            self.global_feature_extractor = get_model('cnn', self.config).to(self.device)
            self.global_feature_extractor.load_state_dict(torch.load(global_path))
            print("Global feature extractor loaded")
        
        # Load client classifiers
        self.client_classifiers = []
        for i in range(self.num_clients):
            client_path = os.path.join(save_path, f'client_{i}_classifier.pth')
            if os.path.exists(client_path):
                classifier = ClassificationHead(self.feature_dim, self.num_classes).to(self.device)
                classifier.load_state_dict(torch.load(client_path))
                self.client_classifiers.append(classifier)
        
        print(f"Loaded {len(self.client_classifiers)} client classifiers")

# Baseline comparison methods
class TraditionalOSFL:
    """Traditional One-Shot Federated Learning baseline"""
    
    def __init__(self, config, train_loaders, test_loaders, device):
        self.config = config
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.device = device
        self.num_clients = len(train_loaders)
        self.num_classes = config['num_classes']
    
    def run(self):
        """Run traditional OSFL"""
        print("Running Traditional One-Shot FL...")
        
        # Stage 1: Each client distills their local dataset (supervised)
        synthetic_datasets = self.supervised_distillation()
        
        # Stage 2: Train global classification model
        global_model = self.train_global_model(synthetic_datasets)
        
        # Evaluate on all clients
        client_accuracies = []
        for client_id in range(self.num_clients):
            acc = self.evaluate_global_model(global_model, self.test_loaders[client_id])
            client_accuracies.append(acc)
        
        return {
            'client_accuracies': client_accuracies,
            'avg_test_acc': np.mean(client_accuracies),
            'std_test_acc': np.std(client_accuracies)
        }
    
    def supervised_distillation(self):
        """Simple supervised dataset distillation"""
        synthetic_datasets = {}
        
        for client_id in range(self.num_clients):
            # Sample random data as synthetic data (simplified)
            train_loader = self.train_loaders[client_id]
            
            x_syn_list = []
            y_syn_list = []
            
            samples_per_class = self.config.get('synthetic_per_client', 100) // self.num_classes
            
            for batch_idx, (data, target) in enumerate(train_loader):
                x_syn_list.append(data[:samples_per_class])
                y_syn_list.append(target[:samples_per_class])
                
                if len(x_syn_list) * samples_per_class >= self.config.get('synthetic_per_client', 100):
                    break
            
            x_syn = torch.cat(x_syn_list, dim=0)[:self.config.get('synthetic_per_client', 100)]
            y_syn = torch.cat(y_syn_list, dim=0)[:self.config.get('synthetic_per_client', 100)]
            
            synthetic_datasets[client_id] = {'x_syn': x_syn, 'y_syn': y_syn}
        
        return synthetic_datasets
    
    def train_global_model(self, synthetic_datasets):
        """Train global classification model"""
        # Aggregate synthetic data
        all_x = []
        all_y = []
        
        for client_id, data in synthetic_datasets.items():
            all_x.append(data['x_syn'])
            all_y.append(data['y_syn'])
        
        x_agg = torch.cat(all_x, dim=0).to(self.device)
        y_agg = torch.cat(all_y, dim=0).to(self.device)
        
        # Create and train model
        from models.networks import FullModel
        model = FullModel(
            self.config['img_channels'], 
            self.config['num_classes'], 
            self.config['feature_dim']
        ).to(self.device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        model.train()
        for epoch in range(1000):
            output = model(x_agg)
            loss = F.cross_entropy(output, y_agg)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return model
    
    def evaluate_global_model(self, model, test_loader):
        """Evaluate global model on test data"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return 100.0 * correct / total
