import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Convolutional block with conv, batchnorm, and relu"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CNN(nn.Module):
    """CNN architecture for feature extraction"""
    def __init__(self, input_channels, feature_dim=512):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            ConvBlock(input_channels, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2, 2),
            
            # Second block  
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),
            
            # Third block
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),
            
            # Fourth block
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.feature_dim = feature_dim
        self.fc = nn.Linear(256, feature_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def embed(self, x):
        """Extract features without final FC layer"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class BarlowTwinsEncoder(nn.Module):
    """Barlow Twins encoder for self-supervised learning"""
    def __init__(self, backbone, projection_dim=2048):
        super(BarlowTwinsEncoder, self).__init__()
        self.backbone = backbone
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)  # Assume CIFAR-10 size
            backbone_output = backbone.embed(dummy_input)
            backbone_dim = backbone_output.shape[1]
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, projection_dim)
        )
        
    def forward(self, x):
        features = self.backbone.embed(x)
        projections = self.projection_head(features)
        return projections

class ClassificationHead(nn.Module):
    """Classification head for personalization"""
    def __init__(self, feature_dim, num_classes, dropout=0.5):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x

class FullModel(nn.Module):
    """Complete model with feature extractor and classification head"""
    def __init__(self, input_channels, num_classes, feature_dim=512):
        super(FullModel, self).__init__()
        self.feature_extractor = CNN(input_channels, feature_dim)
        self.classifier = ClassificationHead(feature_dim, num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x):
        return self.feature_extractor(x)

class ResNetBlock(nn.Module):
    """Basic ResNet block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    """ResNet-18 architecture"""
    def __init__(self, input_channels, feature_dim=512):
        super(ResNet18, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, feature_dim)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResNetBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def embed(self, x):
        """Extract features before final FC layer"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

def get_model(model_name, config):
    """Get model based on configuration"""
    input_channels = config['img_channels']
    feature_dim = config['feature_dim']
    num_classes = config['num_classes']
    
    if model_name.lower() == 'cnn':
        return CNN(input_channels, feature_dim)
    elif model_name.lower() == 'resnet':
        return ResNet18(input_channels, feature_dim)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_ssl_model(backbone, config):
    """Get self-supervised learning model"""
    ssl_feature_dim = config['ssl_feature_dim']
    return BarlowTwinsEncoder(backbone, ssl_feature_dim)

def barlow_twins_loss(z1, z2, lambda_param=0.005):
    """Barlow Twins loss function"""
    batch_size = z1.shape[0]
    feature_dim = z1.shape[1]
    
    # Normalize features
    z1_norm = (z1 - z1.mean(dim=0)) / z1.std(dim=0)
    z2_norm = (z2 - z2.mean(dim=0)) / z2.std(dim=0)
    
    # Cross-correlation matrix
    c = torch.mm(z1_norm.T, z2_norm) / batch_size
    
    # Loss
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    
    loss = on_diag + lambda_param * off_diag
    return loss

def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# Simple augmentation for self-supervised learning
class SimCLRAugmentation:
    """Simple augmentation for self-supervised learning"""
    def __init__(self, size):
        self.transform = nn.Sequential(
            # Random crop and resize would go here for more complex augmentation
            # For simplicity, we'll use basic transformations
        )
    
    def __call__(self, x):
        # Return two augmented versions of the same image
        x1 = x + 0.1 * torch.randn_like(x)  # Add noise
        x2 = x + 0.1 * torch.randn_like(x)  # Add different noise
        return x1, x2
