import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.networks import get_model, get_ssl_model, barlow_twins_loss, SimCLRAugmentation

class ModelPool:
    """Model pool for KRR-ST algorithm"""
    def __init__(self, config, device):
        self.device = device
        self.num_models = config['num_models']
        self.online_iteration = config['online_iteration']
        self.online_lr = config.get('online_lr', 0.1)
        self.online_wd = config.get('online_wd', 1e-3)
        
        # Model configuration
        self.model_config = config
        
        # Initialize model pool
        self.models = []
        self.optimizers = []
        self.iterations = []
        
        for i in range(self.num_models):
            model = get_model('cnn', config).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.online_lr, 
                                      momentum=0.9, weight_decay=self.online_wd)
            self.models.append(model)
            self.optimizers.append(optimizer)
            self.iterations.append(0)
    
    def sample_model(self):
        """Sample a random model from the pool"""
        idx = np.random.randint(self.num_models)
        return idx, self.models[idx]
    
    def update_model(self, idx, x_syn, y_syn):
        """Update model in the pool"""
        model = self.models[idx]
        optimizer = self.optimizers[idx]
        
        # Reset model if it has trained for too long
        if self.iterations[idx] >= self.online_iteration:
            # Re-initialize model
            self.models[idx] = get_model('cnn', self.model_config).to(self.device)
            self.optimizers[idx] = torch.optim.SGD(
                self.models[idx].parameters(), 
                lr=self.online_lr, momentum=0.9, weight_decay=self.online_wd
            )
            self.iterations[idx] = 0
            model = self.models[idx]
            optimizer = self.optimizers[idx]
        
        # Train for one step
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(x_syn)
        loss = F.mse_loss(pred, y_syn)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        self.iterations[idx] += 1
        
        return loss.item()

class KRRSTDistiller:
    """KRR-ST (Kernel Ridge Regression on Self-Supervised Targets) Distiller"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.lambda_reg = config['lambda_reg']
        
        # Learning rates
        self.ssl_lr = config['ssl_lr']
        self.distill_lr = config['distill_lr']
        
        # Training parameters
        self.ssl_epochs = config.get('ssl_epochs', 100)
        self.distill_iterations = config.get('distill_iterations', 1000)
        
        # Augmentation
        self.augmentation = SimCLRAugmentation(config['img_size'])
        
    def train_ssl_target(self, dataloader):
        """Train self-supervised target model using Barlow Twins"""
        
        # Create backbone and SSL model
        backbone = get_model('cnn', self.config).to(self.device)
        ssl_model = get_ssl_model(backbone, self.config).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(ssl_model.parameters(), lr=self.ssl_lr)
        
        ssl_model.train()
        print("Training self-supervised target model...")
        
        for epoch in trange(self.ssl_epochs, desc="SSL Training"):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(self.device)
                
                # Create two augmented views
                x1, x2 = self.augmentation(data)
                
                # Forward pass
                z1 = ssl_model(x1)
                z2 = ssl_model(x2)
                
                # Compute Barlow Twins loss
                loss = barlow_twins_loss(z1, z2)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Limit batches per epoch for efficiency
                if batch_idx >= 10:  # Process only first 10 batches per epoch
                    break
            
            if epoch % 20 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch}, SSL Loss: {avg_loss:.4f}")
        
        # Return the trained backbone (without projection head)
        return ssl_model.backbone
    
    def generate_synthetic_targets(self, target_model, dataloader, num_synthetic):
        """Generate synthetic targets using the SSL target model"""
        
        target_model.eval()
        synthetic_inputs = []
        synthetic_targets = []
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                targets = target_model(data)
                
                synthetic_inputs.append(data)
                synthetic_targets.append(targets)
                
                # Collect enough synthetic samples
                if len(synthetic_inputs) * data.shape[0] >= num_synthetic:
                    break
        
        # Concatenate and trim to desired size
        x_syn = torch.cat(synthetic_inputs, dim=0)[:num_synthetic]
        y_syn = torch.cat(synthetic_targets, dim=0)[:num_synthetic]
        
        # Make synthetic data learnable parameters
        x_syn = x_syn.clone().detach().requires_grad_(True)
        y_syn = y_syn.clone().detach().requires_grad_(True)
        
        return x_syn, y_syn
    
    def distill_dataset(self, dataloader, num_synthetic):
        """
        Perform KRR-ST dataset distillation
        
        Args:
            dataloader: Training data loader
            num_synthetic: Number of synthetic samples to generate
            
        Returns:
            x_syn: Synthetic input data
            y_syn: Synthetic target representations
        """
        
        print("Starting KRR-ST dataset distillation...")
        
        # Step 1: Train self-supervised target model
        target_model = self.train_ssl_target(dataloader)
        target_model.eval()
        
        # Step 2: Generate initial synthetic data
        x_syn, y_syn = self.generate_synthetic_targets(target_model, dataloader, num_synthetic)
        
        # Step 3: Initialize model pool
        model_pool = ModelPool(self.config, self.device)
        
        # Step 4: Optimize synthetic data
        optimizer = torch.optim.Adam([x_syn, y_syn], lr=self.distill_lr)
        
        print("Optimizing synthetic data...")
        
        for iteration in trange(self.distill_iterations, desc="Distillation"):
            optimizer.zero_grad()
            
            # Sample model from pool
            model_idx, model = model_pool.sample_model()
            model.eval()
            
            # Sample real data batch
            try:
                real_data, _ = next(iter(dataloader))
            except:
                dataloader_iter = iter(dataloader)
                real_data, _ = next(dataloader_iter)
            
            real_data = real_data.to(self.device)
            
            # Get target representations for real data
            with torch.no_grad():
                y_real = target_model(real_data)
            
            # Extract features using the sampled model
            f_syn = model.embed(x_syn)
            f_syn = torch.cat([f_syn, torch.ones(f_syn.shape[0], 1).to(self.device)], dim=1)
            
            with torch.no_grad():
                f_real = model.embed(real_data)
                f_real = torch.cat([f_real, torch.ones(f_real.shape[0], 1).to(self.device)], dim=1)
            
            # Compute kernel matrices
            K_real_syn = torch.mm(f_real, f_syn.t())
            K_syn_syn = torch.mm(f_syn, f_syn.t())
            
            # Regularization
            lambda_eye = self.lambda_reg * torch.trace(K_syn_syn.detach()) * torch.eye(K_syn_syn.shape[0]).to(self.device)
            
            # Solve for optimal weights using KRR
            try:
                weights = torch.linalg.solve(K_syn_syn + lambda_eye, y_syn)
                pred_real = torch.mm(K_real_syn, weights)
            except:
                # Fallback to pseudoinverse if singular
                weights = torch.pinverse(K_syn_syn + lambda_eye) @ y_syn
                pred_real = K_real_syn @ weights
            
            # Compute loss
            loss = F.mse_loss(pred_real, y_real)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update model pool
            model_pool.update_model(model_idx, x_syn.detach(), y_syn.detach())
            
            # Log progress
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item():.6f}")
        
        print("Dataset distillation completed!")
        
        return x_syn.detach(), y_syn.detach()
    
    def run(self, dataloader, num_synthetic):
        """Run the complete KRR-ST algorithm"""
        x_syn, y_syn = self.distill_dataset(dataloader, num_synthetic)
        
        # Ensure data is on CPU for transmission (simulating federated setting)
        x_syn = x_syn.cpu()
        y_syn = y_syn.cpu()
        
        return x_syn, y_syn

# Utility functions for synthetic data optimization
def match_loss(x_syn, y_syn, x_real, y_real, model, loss_fn=F.mse_loss):
    """Compute matching loss between synthetic and real data"""
    
    # Forward pass on synthetic data
    pred_syn = model(x_syn)
    
    # Forward pass on real data  
    with torch.no_grad():
        pred_real = model(x_real)
    
    # Compute loss
    loss = loss_fn(pred_syn, pred_real)
    
    return loss

def gradient_matching_loss(x_syn, y_syn, x_real, y_real, model):
    """Compute gradient matching loss"""
    
    # Get gradients from synthetic data
    pred_syn = model(x_syn)
    loss_syn = F.mse_loss(pred_syn, y_syn)
    grad_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)
    
    # Get gradients from real data  
    pred_real = model(x_real)
    loss_real = F.mse_loss(pred_real, y_real)
    grad_real = torch.autograd.grad(loss_real, model.parameters(), retain_graph=False)
    
    # Compute gradient matching loss
    loss = 0
    for g_syn, g_real in zip(grad_syn, grad_real):
        loss += F.mse_loss(g_syn, g_real.detach())
    
    return loss
