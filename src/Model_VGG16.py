# Using pretrained VGG16 model and adding custom layers

import torch
import torch.nn as nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

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
        self.conv3 = nn.Conv2d(384, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)  # Output 3 channels
        self.conv_final = nn.Conv2d(192, 3, kernel_size=3, padding=1)  # Final conv layer after concatenation

        self.relu = nn.ReLU(inplace=True)
        
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Forward pass through the VGG16 base layers
        x = self.base_features(x)

        # Resize or interpolate the output to match the desired output size
        x = self.upsample(x)

        # Apply intermediate convolutional layers
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
    
        x3 = self.relu(self.conv3(cat1))
        x4 = self.relu(self.conv4(x3))
    
        # Concatenate the intermediate features
        cat2 = torch.cat((x1, x2, x3, x4), 1)[:,:192,:,:]
        
        # Apply the final convolutional layer
        x_final = self.conv_final(cat2)

        # Apply ReLU and the subsequent task
        k = self.relu(x_final)
        output = k * x - k + self.b

        return x_final
