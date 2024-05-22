# Preprocess for AODNet Model

from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2.ximgproc

class Preprocess:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        transformed_image = self.transform(image)

        # Gamma Correction
        gamma_corrected_image = transforms.functional.adjust_gamma(transformed_image, 2.2)

        # Avoid division by zero and use integer types for precise results
        min_val = gamma_corrected_image.min().float()  # Convert to float for division
        max_val = gamma_corrected_image.max().float()
        if max_val == min_val:  # Handle edge case (constant image)
            return transformed_image
        stretched_image = (gamma_corrected_image - min_val) / (max_val - min_val) # min-max normalisation

        # Use integer types for filtering
        stretched_image_np = stretched_image.permute(1, 2, 0).to(torch.float32).numpy()
        guided_filter = cv2.ximgproc.createGuidedFilter(
            guide=stretched_image_np, radius=3, eps=0.01
        )
        filtered_image = guided_filter.filter(src=stretched_image_np)

        return torch.from_numpy(filtered_image).permute(2, 0, 1)