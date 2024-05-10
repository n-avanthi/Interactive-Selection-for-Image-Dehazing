import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from Model_AODNet import AODnet
import numpy as np

device = torch.device("cpu")
model = AODnet().to(device)

# checkpoint_path = "C:\\avanthi\\college\\Image Dehazing\\Interactive-Selection-for-Image-Dehazing\\saved_models\\49_best_model_2024-03-03_01-58-26.pth\\199_best_model_2024-03-04_05-36-54.pth\\AOD_199.pth"
checkpoint_path = "C:\\avanthi\\college\\Image Dehazing\\Interactive-Selection-for-Image-Dehazing\\saved_models\\49_best_model_2024-03-03_01-58-26.pth\\AOD_49.pth"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

class DehazingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Dehazing")
        self.geometry("800x600")

        self.canvas = tk.Canvas(self, width=600, height=400)
        self.canvas.pack()

        self.load_button = tk.Button(self, text="Upload Image", command=self.load_image)
        self.load_button.pack(pady=10)

        self.dehaze_button = tk.Button(self, text="Dehaze", command=self.dehaze_image, state=tk.DISABLED)
        self.dehaze_button.pack(pady=10)

        self.original_image = None
        self.roi = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = Image.open(file_path)
            self.photo = ImageTk.PhotoImage(self.original_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.bind("<ButtonPress-1>", self.start_roi)
            self.canvas.bind("<B1-Motion>", self.update_roi)
            self.canvas.bind("<ButtonRelease-1>", self.end_roi)
            self.dehaze_button.config(state=tk.NORMAL)
            self.image_bounds = (0, 0, self.original_image.width, self.original_image.height)

    def start_roi(self, event):
        x, y = event.x, event.y
        x = max(x, self.image_bounds[0])  # Clamp x coordinate within image bounds
        x = min(x, self.image_bounds[2])
        y = max(y, self.image_bounds[1])  # Clamp y coordinate within image bounds
        y = min(y, self.image_bounds[3])
        self.roi = (x, y, x, y)

    def update_roi(self, event):
        if self.roi:
            self.canvas.delete("roi")
            x, y = event.x, event.y
            x = max(x, self.image_bounds[0])  # Clamp x coordinate within image bounds
            x = min(x, self.image_bounds[2])
            y = max(y, self.image_bounds[1])  # Clamp y coordinate within image bounds
            y = min(y, self.image_bounds[3])
            self.roi = (self.roi[0], self.roi[1], x, y)
            self.canvas.create_rectangle(self.roi, outline="red", tags="roi")

    def end_roi(self, event):
        if self.roi:
            x, y = event.x, event.y
            x = max(x, self.image_bounds[0])  # Clamp x coordinate within image bounds
            x = min(x, self.image_bounds[2])
            y = max(y, self.image_bounds[1])  # Clamp y coordinate within image bounds
            y = min(y, self.image_bounds[3])
            self.roi = (self.roi[0], self.roi[1], x, y)
            self.canvas.delete("roi")
            roi_image = self.original_image.crop(self.roi)
            self.photo = ImageTk.PhotoImage(roi_image)
            self.canvas.delete("all")  # Clear the canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def dehaze_image(self):
        if self.original_image is None:
            print("Please load an image first.")
            return

        if self.roi is None:
            print("Please select a region of interest.")
            return

        roi_image = self.original_image.crop(self.roi)
        roi_image_np = np.array(roi_image.resize((512, 512)))  # Convert Pillow Image to NumPy array
        roi_tensor = torch.Tensor(roi_image_np).permute(2, 0, 1).unsqueeze(0).to(device)
        dehazed_tensor = model(roi_tensor)
        dehazed_tensor = dehazed_tensor.detach()  # Detach the tensor from the computational graph
        dehazed_image_np = dehazed_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        dehazed_image = Image.fromarray((dehazed_image_np * 255).astype('uint8'))
        self.photo_dehazed = ImageTk.PhotoImage(dehazed_image)
        self.canvas.delete("all")  # Clear the canvas
        self.canvas.create_image(300, 0, anchor=tk.NW, image=self.photo_dehazed)

if __name__ == "__main__":
    app = DehazingApp()
    app.mainloop()