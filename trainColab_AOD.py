# Training AODNet model

import os
import datetime
from PIL import Image
import pathlib
# from DehazingDataset_AODNet import DatasetType, DehazingDataset
# from Model_AODNet import AODnet
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tu_data
import torchvision.transforms as transforms
import torchvision.transforms.functional as tv_functional
from torchmetrics.image import StructuralSimilarityIndexMeasure
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2.ximgproc
import numpy as np

def GetProjectDir() -> pathlib.Path:
    # return pathlib.Path(__file__).parent.parent
    return "/content/drive/MyDrive/"

def Preprocess(image: Image.Image) -> torch.Tensor:
    # PIL images are converted to PyTorch tensor which is a multidimensional array
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
        ]
    )
    transformedImage = transform(image)

    # Gamma Correction involves adjusting brightness and contrast of an image using its pixel values
    gammaCorrectedImage = transforms.functional.adjust_gamma(transformedImage, 2.2)

    # Histogram Stretching improves the contrast by spreading out its pixel intensity values over a wider range
    min_val = gammaCorrectedImage.min()
    max_val = gammaCorrectedImage.max()
    if max_val == min_val:  # Handle edge case (constant image)
            return transformedImage
    stretchedImage = (gammaCorrectedImage - min_val) / (max_val - min_val)

    # Guided Filtering improves the visual quality of an image while preserving important details and edges
    stretched_image_np = stretchedImage.permute(1, 2, 0).to(torch.float32).numpy()
    gFilter = cv2.ximgproc.createGuidedFilter(guide=stretched_image_np, radius=3, eps=0.01)
    filteredImage = gFilter.filter(src=stretched_image_np)
    return torch.from_numpy(filteredImage).permute(2, 0, 1)

def VEF(input_image, output_image, target_image):

    # Convert images to grayscale
    input_gray = torch.mean(input_image, dim=1, keepdim=True)
    output_gray = torch.mean(output_image, dim=1, keepdim=True)
    target_gray = torch.mean(target_image, dim=1, keepdim=True)

    # Calculate image differences
    diff_out_target = torch.abs(output_gray - target_gray)
    diff_in_target = torch.abs(input_gray - target_gray)

    vef = 1 - (torch.mean(diff_out_target) / torch.mean(diff_in_target))

    return vef.item()

# Saves the state of a trained neural network model
def save_model(epoch, path, net, optimizer, net_name):
    if not os.path.exists(os.path.join(path, net_name)):
         os.mkdir(os.path.join(path, net_name))
    torch.save(
        {"epoch": epoch, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict()},
        f=os.path.join(path, net_name, "{}_{}.pth".format("AOD", epoch)),
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")

    datasetPath = GetProjectDir() + "dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k"
    trainingDataset = DehazingDataset(dehazingDatasetPath=datasetPath, _type=DatasetType.Train, transformFn=Preprocess, verbose=False)
    validationDataset = DehazingDataset(dehazingDatasetPath=datasetPath, _type=DatasetType.Validation, transformFn=Preprocess, verbose=False) # Verbosity level indicates level of additional information provided during execution

    # Batch size is the number of samples used in one iteration of training
    batch_size = 2

    trainingDataLoader = tu_data.DataLoader(trainingDataset, batch_size=batch_size, shuffle=True, num_workers=3)
    validationDataLoader = tu_data.DataLoader(validationDataset, batch_size=batch_size, shuffle=True, num_workers=3) # Num_workers controls the number of subprocesses to use for data loading

    print(len(trainingDataset), len(validationDataset))

    model = AODnet().to(device)
    print(model) # Prints the model summary like  architecture, number of parameters, and layer configurations

    best_ssim = 0.0
    best_vef = 0.0
    avg_psnr = 0.0
    criterion = nn.MSELoss().to(device=device) # Loss function is initialised and MSE is used
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # Adam optimizer is initialised and learning rate is set to 10^(-3)

    EPOCHS = 100 # Training data will be passed through the model for training 100 times
    patience = 5
    early_stopping_counter = 0
    train_number = len(trainingDataLoader)

    print("Started Training...")
    model.train()
    for epoch in range(EPOCHS):
        for step, (haze_image, ori_image) in enumerate(trainingDataLoader):
            try:
                ori_image, haze_image = ori_image.to(device), haze_image.to(device)
                dehaze_image = model(haze_image)
                loss = criterion(dehaze_image, ori_image)
                optimizer.zero_grad() # Clears the gradients of all optimized parameters to prevent gradients from previous iterations from affecting the parameter updates
                loss.backward() # Computes gradients of the loss with respect to the model parameters (backpropogation)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Used to clip gradients to prevent the exploding gradient problem during training
                optimizer.step() # Updates the model parameters using computed gradients and optimization algorithm

                if (step + 1) % 10 == 0 or step + 1 == train_number:
                    print(
                      "Epoch: {}/{}  |  Step: {}/{}  |  lr: {:.6f}  | Loss: {:.6f}".format(
                        epoch + 1, EPOCHS, step + 1, train_number, optimizer.param_groups[0]["lr"], loss.item()
                        )
                    )
            except FileNotFoundError as e:
                # Handle missing file error, for example, print a message
                print(f"Error: {e}. Skipping this batch.")

        # -------------------------------------------------------------------
        # Validation loop
        avg_ssim = 0.0
        avg_vef = 0.0
        avg_psnr = 0.0
        total_batches = len(validationDataLoader)
        print("Epoch: {}/{} | Validation Model Saving Images".format(epoch + 1, EPOCHS))
        model.eval() # In evaluation mode the model performs inference without modifying its parameters

        for step, (haze_image, ori_image) in enumerate(validationDataLoader):
            try:
                if step > 10:  # only save image 10 times
                    break
                ori_image, haze_image = ori_image.to(device), haze_image.to(device)
                dehaze_image = model(haze_image)

                # Structural Similarity Index is a score from -1 to 1 that gives the similarity between two images based on luminance similarity, contrast similarity, and structural similarity.
                # Higher SSIM scores indicate better preservation of image structure and visual quality, while lower scores may indicate loss of important details or introduction of artifacts during the dehazing process
                ssim = StructuralSimilarityIndexMeasure().to(device)
                ssim_val = ssim(dehaze_image, ori_image)
                avg_ssim += ssim_val

                vef_val = VEF(dehaze_image, ori_image, haze_image)
                avg_vef += vef_val

                psnr_val = psnr(torch.clamp(dehaze_image, 0, 1).cpu().detach().numpy(), torch.clamp(ori_image, 0, 1).cpu().detach().numpy())
                avg_psnr += psnr_val

                # Convert images to NumPy arrays
                ori_image_np = np.array(torchvision.transforms.ToPILImage()(ori_image[0].cpu()))
                dehaze_image_np = np.array(torchvision.transforms.ToPILImage()(dehaze_image[0].cpu()))

                # torchvision.utils.save_image(
                #     torchvision.utils.make_grid(torch.cat((haze_image, dehaze_image, ori_image), 0), nrow=ori_image.shape[0]),
                #     os.path.join(GetProjectDir() + "output", "{}_{}.jpg".format(epoch + 1, step)),
                # )
            except:
                # Handle missing file error, for example, print a message
                print(f"Error: {e}. Skipping this batch.")

        avg_ssim /= total_batches
        avg_vef /= total_batches
        avg_psnr /= total_batches
        print(f"Avg SSIM for Epoch {epoch + 1}: {avg_ssim}")
        print(f"Avg VEF for Epoch {epoch + 1}: {avg_vef}")
        print(f"Avg PSNR for Epoch {epoch + 1}: {avg_psnr}")
        model.train()

        # Early stopping
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            best_model_path = os.path.join(
              GetProjectDir() + "saved_models",
              "{}_best_model_{}.pth".format(epoch, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} as validation SSIM did not improve for {patience} epochs.")
                break
    if best_model_path:
        save_model(epoch, GetProjectDir() + "saved_models", model, optimizer, best_model_path)
    # Save the best model after all epochs
    if best_model_path:
        save_model(epoch, GetProjectDir() + "saved_models", model, optimizer, best_model_path)