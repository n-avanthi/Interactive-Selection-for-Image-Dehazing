# UI for AODNet model with interactive selection

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
        self.bbox_start = None
        self.temp_rectangle = None
        self.dehazed_image = None  # Store the dehazed image for display

        # UI Elements
        self.canvas = tk.Canvas(root, width=512, height=512, highlightthickness=5, highlightbackground="black")
        self.canvas.pack()

        self.upload_button = tk.Button(root, text="Upload Image", command=self.open_image)
        self.upload_button.pack()

        self.dehaze_button = tk.Button(root, text="Dehaze Selected Area", command=self.dehaze_selected_area)
        self.dehaze_button.pack()

        # Load AODnet model (consider adding error handling for loading issues)
        try:
            self.aodnet_model = self.load_aodnet_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            # Handle loading error (e.g., display error message, disable dehazing button)

        # Event bindings for drawing bounding box
        self.canvas.bind("<Button-1>", self.draw_bbox_start)
        self.canvas.bind("<B1-Motion>", self.draw_bbox_update)
        self.canvas.bind("<ButtonRelease-1>", self.draw_bbox_release)

    def open_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = Image.open(self.image_path)
            img = self.original_image.copy()  # Make a copy
            img.thumbnail((600, 600))  # Resize for display
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.img = img_tk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.dehazed_image = None  # Reset dehazed image when opening a new image


    def draw_bbox_start(self, event):
        self.bbox_start = (event.x, event.y)
        if self.temp_rectangle:
            self.canvas.delete(self.temp_rectangle)
        self.temp_rectangle = self.canvas.create_rectangle(0, 0, 0, 0, outline='red')

    def draw_bbox_update(self, event):
        x0, y0 = self.bbox_start
        x1, y1 = event.x, event.y
        self.canvas.coords(self.temp_rectangle, x0, y0, x1, y1)

    def draw_bbox_release(self, event):
        x0, y0 = self.bbox_start
        x1, y1 = event.x, event.y

        # Ensure coordinates are within image bounds
        max_x = self.original_image.width - 1
        max_y = self.original_image.height - 1
        x0 = max(0, min(x0, max_x))  # Clamp x0 between 0 and max_x
        y0 = max(0, min(y0, max_y))  # Clamp y0 between 0 and max_y
        x1 = max(0, min(x1, max_x))  # Clamp x1 between 0 and max_x
        y1 = max(0, min(y1, max_y))  # Clamp y1 between 0 and max_y

        self.bbox_coordinates = (x0, y0, x1, y1)

        # Dehaze the selected area if a dehazed image doesn't already exist
        if not self.dehazed_image:
            self.dehaze_selected_area()


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

    def dehaze_selected_area(self):
        if hasattr(self, 'bbox_coordinates'):
            x0, y0, x1, y1 = self.bbox_coordinates
            # Extract the selected area from the original image
            selected_area = self.original_image.crop((x0, y0, x1, y1))

            # Get the selected area size
            selected_area_width, selected_area_height = selected_area.size

            # Create an instance of the Preprocess class
            preprocessor = Preprocess()

            # Call the preprocess method on the selected area
            preprocessed_segmented_image = preprocessor.preprocess(selected_area)

            # Convert the preprocessed image to PIL format
            dehazed_img_pil = transforms.ToPILImage()(preprocessed_segmented_image)

            # Resize the dehazed image to match the selected area size
            dehazed_img_resized = dehazed_img_pil.resize((selected_area_width, selected_area_height), Image.LANCZOS)

            # Convert the resized image to tkinter format
            dehazed_img_tk = ImageTk.PhotoImage(dehazed_img_resized)

            # Display the dehazed result on the canvas
            self.canvas.create_image(x0, y0, anchor=tk.NW, image=dehazed_img_tk, tags="dehazed_region")
            self.canvas.img = dehazed_img_tk

    def draw_bbox_release(self, event):
        x0, y0 = self.bbox_start
        x1, y1 = event.x, event.y

        # Ensure coordinates are within image bounds
        max_x = self.original_image.width - 1
        max_y = self.original_image.height - 1
        x0 = max(0, min(x0, max_x))  # Clamp x0 between 0 and max_x
        y0 = max(0, min(y0, max_y))  # Clamp y0 between 0 and max_y
        x1 = max(0, min(x1, max_x))  # Clamp x1 between 0 and max_x
        y1 = max(0, min(y1, max_y))  # Clamp y1 between 0 and max_y

        self.bbox_coordinates = (x0, y0, x1, y1)

        # Remove any previously displayed dehazed region
        self.canvas.delete("dehazed_region")

        # Dehaze the selected area if a dehazed image doesn't already exist
        if not self.dehazed_image:
            self.dehaze_selected_area()
    def display_dehazed_region(self, dehazed_region):
        # Display the dehazed result
        dehazed_tk = ImageTk.PhotoImage(dehazed_region)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=dehazed_tk)
        self.canvas.img = dehazed_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDehazerApp(root)
    root.mainloop()
