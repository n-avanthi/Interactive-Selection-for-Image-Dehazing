# Using pretrained ResNet18 model and adding custom layers

import torch
import torch.nn as nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Load a pre-trained ResNet model as the base
        resnet = models.resnet18(pretrained=True)

        # Freeze all parameters of the base ResNet layers
        for param in resnet.parameters():
            param.requires_grad = False

        # Extract the last convolutional layer of the ResNet model
        self.base_layers = nn.Sequential(*list(resnet.children())[:-2])
        self.conv5 = resnet.layer4[-1]  # Get the last convolutional layer from the ResNet model

        # Define additional layers for the DCP model
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(384, 64, kernel_size=3, padding=1)  # Adjusted number of input channels
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)  # Output 3 channels
        self.conv_final = nn.Conv2d(451, 3, kernel_size=3, padding=1)  # Adjusted number of input channels

        self.relu = nn.ReLU(inplace=True)
        
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # Forward pass through the ResNet base layers
        x_base = self.base_layers(x)
        x_resnet = self.conv5(x_base)

        # Upsample the feature maps to the desired output size
        x_upsampled = self.upsample(x_resnet)

        # Apply intermediate convolutional layers
        x1 = self.relu(self.conv1(x_upsampled))
        x2 = self.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
    
        x3 = self.relu(self.conv3(cat1))
        x4 = self.relu(self.conv4(x3))
    
        # Concatenate the intermediate features
        cat2 = torch.cat((x1, x2, x3, x4), 1)
        
        #print("Shape of cat2:", cat2.shape)
        
        # Apply the final convolutional layer
        x_final = self.conv_final(cat2)
        
        # Apply ReLU and the subsequent task
        k = self.relu(x_final)
        output = k * x - k + self.b

        return x_final
