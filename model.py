import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import SwinForImageClassification

class ParallelResNetSwin(nn.Module):
    def __init__(self, num_classes=4):
        super(ParallelResNetSwin, self).__init__()
        # ResNet
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet_features = self.resnet.fc.in_features  # Store the number of features
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer
        # Swin Transformer
        model_name = "microsoft/swin-base-patch4-window7-224-in22k"
        self.swin = SwinForImageClassification.from_pretrained(model_name)
        self.swin.classifier = nn.Identity()  # Remove the final classification layer
        # Combine features
        combined_features = self.resnet_features + self.swin.config.hidden_size
        self.classifier = nn.Linear(combined_features, num_classes)

    def forward(self, x):
        # ResNet forward pass
        resnet_features = self.resnet(x)
        # Swin Transformer forward pass
        swin_features = self.swin(x).logits
        # Combine features
        combined = torch.cat((resnet_features, swin_features), dim=1)
        # Final classification
        output = self.classifier(combined)
        return output