# Defining dataset for Resnet18 and VGG16 Models with size (224, 224)

import os
import pathlib
from enum import Enum
import torch
from torch.utils.data import Dataset
from PIL import Image

class DatasetType(Enum):
    Train = 0,
    Test = 1,
    Validation = 2

    def ToString(self) -> str:
        if self == DatasetType.Train:
            return 'train'
        elif self == DatasetType.Test:
            return 'test'
        elif self == DatasetType.Validation:
            return 'val'

class DehazingDataset(Dataset):
    def __init__(self, dehazingDatasetPath: pathlib.Path, _type: DatasetType, transformFn=None, verbose: bool = False):
        self.__DehazingDatasetPath = dehazingDatasetPath
        self.__TransformFn = transformFn

        self.__HazyImages = []
        self.__ClearImages = []

        for variant in ('Haze1k_thin', 'Haze1k_moderate', 'Haze1k_thick'):
            inputPath = self.__DehazingDatasetPath / variant / 'dataset' / _type.ToString() / 'input'
            targetPath = self.__DehazingDatasetPath / variant / 'dataset' / _type.ToString() / 'target'

            self.__HazyImages += [inputPath / filename for filename in sorted(os.listdir(inputPath)) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
            self.__ClearImages += [targetPath / filename for filename in sorted(os.listdir(targetPath)) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

        # Filtering the mismatching (input, target) image pair
        assert len(self.__HazyImages) == len(self.__ClearImages)
        for hazyPath, clearPath in zip(self.__HazyImages, self.__ClearImages):
            hazyImage = Image.open(hazyPath)
            clearImage = Image.open(clearPath)
            if hazyImage.size != clearImage.size:
                self.__HazyImages.remove(hazyPath)
                self.__ClearImages.remove(clearPath)
            elif verbose:
                print(hazyPath)
                print(clearPath)

        self.__Size = len(self.__HazyImages)

    def __len__(self):
        return self.__Size

    def __getitem__(self, index) -> torch.Tensor:
        hazyImage = None
        clearImage = None
        try:
            hazyImage = torch.Tensor(self.__TransformFn(Image.open(self.__HazyImages[index]).convert('RGB')))
            clearImage = torch.Tensor(self.__TransformFn(Image.open(self.__ClearImages[index]).convert('RGB')))
        except OSError:
            print(f'Error Loading: {self.__HazyImages[index]}')
            print(f'Error Loading: {self.__ClearImages[index]}')

            placeholder_image = torch.zeros((3, 224, 224), dtype=torch.float32)
            return placeholder_image, placeholder_image

        return hazyImage, clearImage