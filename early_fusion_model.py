
import torch
import torch.nn as nn
from torchvision import models

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes, fine_tune=True):
        super(EarlyFusionModel, self).__init__()
        
        # Load Pre-trained ResNet18
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)
        
        # Modify the first layer to accept 6 channels (3 for hand + 3 for iris)
        # Original: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=6, 
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Smart Initialization for the new 6-channel layer
        # We copy the pre-trained weights into both the first 3 channels and the last 3 channels.
        # We scale by 0.5 so that the expected magnitude of the output remains similar
        # (since we are summing twice as many inputs).
        with torch.no_grad():
            self.backbone.conv1.weight[:, :3] = original_conv1.weight * 0.5
            self.backbone.conv1.weight[:, 3:] = original_conv1.weight * 0.5
            
        # Modify the fully connected layer for classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Freeze hidden layers if not fine-tuning (Optional, usually we fine-tune all for early fusion)
        # Since we modified the first layer significantly, freezing everything else might not be ideal immediately,
        # but standard transfer learning often freezes early layers. 
        # However, for Early Fusion where the input statistics change drastically, full fine-tuning is safer.
        # But to respect the request's implication of "similar params", I'll leave the default fine_tune logic.
        if not fine_tune:
            for name, param in self.backbone.named_parameters():
                # Don't freeze the modified first layer or the new fc
                if "conv1" not in name and "fc" not in name:
                    param.requires_grad = False

    def forward(self, hand_img, iris_img):
        # Concatenate inputs along channel dimension (B, 3, H, W) -> (B, 6, H, W)
        x = torch.cat((hand_img, iris_img), dim=1)
        
        # Forward through backbone
        output = self.backbone(x)
        
        return output
