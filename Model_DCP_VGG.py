# Using pretrained VGG16 model as base and adding layers from DCP model

import torch
import torch.nn as nn
import torchvision.models as models

class DCPModel(nn.Module):
    def __init__(self):
        super(DCPModel, self).__init__()

        # Load a pre-trained VGG16 model as the base
        vgg16_model = models.vgg16(pretrained=True)

        # Freeze all parameters of the base VGG16 layers
        for param in vgg16_model.parameters():
            param.requires_grad = False

        # Extract the features from the VGG16 model
        self.base_features = vgg16_model.features

        # Define additional layers for the DCP model
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)  # Output 3 channels

        self.relu = nn.ReLU(inplace=True)
        
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Forward pass through the VGG16 base layers
        x = self.base_features(x)

        # Apply additional convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)  # Output 3 channels

        # Resize or interpolate the output to match the desired output size
        x = self.upsample(x)

        return x
