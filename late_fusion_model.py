
import torch
import torch.nn as nn
from torchvision import models

class LateFusionModel(nn.Module):
    def __init__(self, num_classes, fine_tune=True):
        super(LateFusionModel, self).__init__()
        
        # Hand Branch
        weights = models.ResNet18_Weights.DEFAULT
        self.hand_backbone = models.resnet18(weights=weights)
        self.hand_feature_dim = self.hand_backbone.fc.in_features
        # Remove the classification head
        self.hand_backbone.fc = nn.Identity()
        
        # Iris Branch
        self.iris_backbone = models.resnet18(weights=weights)
        self.iris_feature_dim = self.iris_backbone.fc.in_features
        self.iris_backbone.fc = nn.Identity()
        
        # Helper to freeze weights if needed
        if not fine_tune:
            for param in self.hand_backbone.parameters():
                param.requires_grad = False
            for param in self.iris_backbone.parameters():
                param.requires_grad = False
        
        # Fusion Head
        # Concatenate features: 512 + 512 = 1024
        self.fusion_dim = self.hand_feature_dim + self.iris_feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, hand_img, iris_img):
        # Extract features
        hand_features = self.hand_backbone(hand_img)
        iris_features = self.iris_backbone(iris_img)
        
        # Flatten if necessary (ResNet usually performs GAP so output is [B, 512])
        # Identity layer just passes it through, so we are good.
        
        # Fuse (Concatenate)
        fused_features = torch.cat((hand_features, iris_features), dim=1)
        
        # Classify
        output = self.classifier(fused_features)
        
        return output
