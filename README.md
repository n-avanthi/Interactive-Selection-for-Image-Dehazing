# Interactive-Selection-for-Image-Dehazing
Image dehazing improves visibility and enhances image quality in hazy or polluted conditions by restoring the original radiance of the scene, resulting in clearer and visually appealing images for better interpretation.

## Requirements

- python 3.10
- Pillow 10.0.0
- torch 2.2.1+cu121
- torchvision 0.17.1+cu121
- torchmetrics 1.3.1
- scikit-image 0.22.0
 - tk 0.1.0

## Models

#### 1. AODNet
AODNet is a deep learning architecture specifically designed for dehazing and is implemented as a neural network. Adaptive Moment Estimation (Adam) Optimiser is the optimization algorithm used for training. In each iteration, it computes the gradient of the loss function with respect to the parameters.

#### 2. VGG16
VGG16 is a convolutional neural network architecture that is widely used in computer vision tasks. Here VGG16 model was used as a base, and four custom convolution layers were added at the end. The outputs of these custom layers were concatenated to give the final output.

#### 3. ResNet18
ResNet18 is a convolutional neural network architecture known for its residual connections, which help alleviate the vanishing gradient problem. Similar to the VGG16 model, it was used as a base, and four custom convolution layers were added. The same function was performed as with the VGG16 model.

## Dataset
A dataset consisting of Multispectral Satellite Images was used which were captured by an ISRO satellite.
- It contains 1200 images.
- Images are of size 512x512 pixels.
- It contains 960 training images, 135 testing images and 105 validation images.

To download and set up the database
```bash
$ python setup_dataset.py 
```


## Train the model
To train the AODNet model

```bash
# preprocessing the images
$ python Preprocess.py
```
```bash
# loading and filtering image pairs from dataset
$ python DehazingDataset.py
```
```bash
# defining the AODnet model(5 layers)
$ python Model_AODNet.py
# defining the AODnet model(7 layers)
$ python Model_AODNet_1.py
```
```bash
# training the model
$ python train_AOD.py
```

To train the ResNet18 model

```bash
# preprocessing the images
$ python Preprocess_1.py
```
```bash
# loading and filtering image pairs from dataset
$ python DehazingDataset_1.py
```
```bash
# defining the model
$ python Model_ResNet18.py
```
```bash
# training the model
$ python train_resnet18.py
```

To train the VGG16 model

```bash
# preprocessing the images
$ python Preprocess_1.py
```
```bash
# loading and filtering image pairs from dataset
$ python DehazingDataset_1.py
```
```bash
# defining the model
$ python Model_VGG16.py
```
```bash
# training the model
$ python train_vgg16.py
```

## 4. test the model
Test the AODNet model
```bash
# include path to saved model in model_path
# change the preprocessing as required by the model(AODNet is default)
$ python ui.py
# to test the interactive model
$ python ui-interactive.py
```

