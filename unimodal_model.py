
import torch
import torch.nn as nn
from torchvision import models

class UnimodalModel(nn.Module):
    """
    A single-modality biometric classifier using ResNet18.
    Can be used for Hand-only or Iris-only classification.
    """
    def __init__(self, num_classes, fine_tune=True):
        super(UnimodalModel, self).__init__()
        
        # Load Pre-trained ResNet18
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)
        
        # Modify the fully connected layer for classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Freeze backbone if not fine-tuning
        if not fine_tune:
            for name, param in self.backbone.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)
