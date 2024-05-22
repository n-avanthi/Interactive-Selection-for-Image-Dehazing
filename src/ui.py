# UI for AODNet model

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import torch
from Model_AODNet import AODnet  # Assuming this is your custom model class
from torchvision.io import read_image
import io
from Preprocess import Preprocess  # Assuming this is your custom preprocessing class


class ImageDehazerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Dehazing App")

        # Variables
        self.image_path = None
        self.dehazed_image = None  # Store the dehazed image for display

        # UI Elements
        self.canvas = tk.Canvas(root, width=600, height=600)
        self.canvas.pack()

        self.upload_button = tk.Button(root, text="Upload Image", command=self.open_image)
        self.upload_button.pack()

        self.dehaze_button = tk.Button(root, text="Dehaze Image", command=self.dehaze_image)
        self.dehaze_button.pack()

        # Load AODnet model (consider adding error handling for loading issues)
        try:
            self.aodnet_model = self.load_aodnet_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            # Handle loading error (e.g., display error message, disable dehazing button)

    def open_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = Image.open(self.image_path)
            img = self.original_image.copy()  # Make a copy
            img.thumbnail((512, 512))  # Resize for display
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.img = img_tk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.dehazed_image = None  # Reset dehazed image when opening a new image

    def load_aodnet_model(self):
        # Specify the path to your model weights
        model_path = "D:\\Avanthi\\personal\\Interactive-Selection-for-Image-Dehazing\\saved_models\\AOD_199.pth"

        # Load the model weights into memory using io.BytesIO
        with open(model_path, 'rb') as f:
            buffer = io.BytesIO(f.read())

        # Instantiate an AODnet model
        model = AODnet()

        # Load the model weights from the buffer
        checkpoint = torch.load(buffer)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()  # Set the model to evaluation mode

        return model

    def dehaze_image(self):
        if self.original_image is not None:
            # Create an instance of the Preprocess class
            preprocessor = Preprocess()

            # Get the original image size
            original_width, original_height = self.original_image.size

            # Call the preprocess method on the entire image
            preprocessed_image = preprocessor.preprocess(self.original_image)

            # Convert the preprocessed image to PIL format
            dehazed_img_pil = transforms.ToPILImage()(preprocessed_image)

            # Resize the dehazed image to match the original size
            dehazed_img_resized = dehazed_img_pil.resize((original_width, original_height), Image.LANCZOS)

            # Convert the resized image to tkinter format
            dehazed_img_tk = ImageTk.PhotoImage(dehazed_img_resized)

            # Update canvas size to match the dimensions of the resized image
            self.canvas.config(width=original_width, height=original_height)
            self.canvas.img = dehazed_img_tk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=dehazed_img_tk)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDehazerApp(root)
    root.mainloop()
