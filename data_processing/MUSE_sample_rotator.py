import sys
sys.path.insert(0, '..')

import tkinter as tk
from tkinter import messagebox, ttk, font
import os
from PIL import Image, ImageTk
from project_settings import MUSE_DATA_FOLDER
import pickle
import numpy as np
from tqdm import tqdm
from tools.utils import cropper, mask_circle
from scipy.ndimage import rotate


class ImageLabeller:
    def __init__(self, master, pickle_files, labels_file):
        self.master = master
        self.pickle_files = pickle_files
        self.labels_file = labels_file
        self.current_image_index = 0
        self.current_angle = 0

        self.mask = 1 - mask_circle(200, 25)

        # Load the default PSF image
        self.PSF_default = np.load(MUSE_DATA_FOLDER + 'PSF_default.npy').mean(axis=0)
        self.PSF_default = 1 + np.log10(np.abs(self.PSF_default))
        self.PSF_default *= 1 - mask_circle(self.PSF_default.shape[0], 25)
        self.PSF_default -= self.PSF_default.min()
        self.PSF_default /= self.PSF_default.max()
        self.PSF_default = 1 - self.PSF_default * 0.35

        self.show_overlay = True

        self.selections = self.initialize_angles()
        self.setup_gui()


    def setup_gui(self):
        self.master.title("Dataset Labeller")

        # Image display setup
        self.image_label = ttk.Label(self.master)
        self.image_label.pack()

        # Navigation buttons
        prev_button = ttk.Button(self.master, text="Previous", command=self.prev_image)
        prev_button.pack(side=tk.LEFT)
        next_button = ttk.Button(self.master, text="Next", command=self.next_image)
        next_button.pack(side=tk.RIGHT)

        # Bind arrow keys for navigation
        self.master.bind('<Left>', lambda e: self.prev_image())
        self.master.bind('<Right>', lambda e: self.next_image())

        # Frame for ID jump functionality
        self.jump_frame = ttk.Frame(self.master)
        self.jump_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Textbox (Entry) for inputting the sample ID
        self.id_entry = ttk.Entry(self.jump_frame)
        self.id_entry.pack(side=tk.LEFT, padx=(0, 10))

        # Button to jump to the sample
        self.jump_button = ttk.Button(self.jump_frame, text="Go to ID", command=self.jump_to_sample)
        self.jump_button.pack(side=tk.LEFT)

        # Bind the Enter key to the jump action as well
        self.id_entry.bind('<Return>', lambda event=None: self.jump_to_sample())

        # Frame for rotation angle functionality
        self.rotation_frame = ttk.Frame(self.master)
        self.rotation_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # Label and Textbox (Entry) for rotation angle
        self.rotation_label = ttk.Label(self.rotation_frame, text="Rotation Angle:")
        self.rotation_label.pack(side=tk.LEFT, padx=(0, 10))

        self.angle_entry = ttk.Entry(self.rotation_frame)
        self.angle_entry.pack(side=tk.LEFT, padx=(0, 10))

        angle_delta = 1  # [deg]

        # Bind arrow keys for angle adjustment
        self.master.bind('<Up>', lambda e: self.change_angle(angle_delta))
        self.master.bind('<Down>', lambda e: self.change_angle(-angle_delta))

        self.angle_entry.bind('<Return>', lambda event=None: self.set_angle())

        self.load_image()
        self.load_labels()


    def initialize_angles(self):
        self.selections = {}
        if os.path.exists(self.labels_file):
            self.load_labels()
            return self.selections
        else:
            angles = {}
            for pickle_file in tqdm(self.pickle_files):
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    if 'Pupil angle' in data['MUSE header data'].keys():
                        image_name = os.path.basename(pickle_file)
                        angles[image_name] = np.round(data['MUSE header data']['Pupil angle'].item(), 2) % 360
                    else:
                        angles[image_name] = 0.0

            self.save_labels(angles)
        return angles


    def jump_to_sample(self):
        # Get the entered sample ID from the textbox
        sample_id = self.id_entry.get()

        # Find the index of the image that matches the sample ID
        for index, path in enumerate(self.pickle_files):
            # Assuming the sample ID is part of the filename, adjust as necessary
            if sample_id == os.path.basename(path).split('_')[0]:
                self.current_image_index = index
                self.load_image()
                break
        else:
            messagebox.showinfo("Info", f"Sample ID {sample_id} not found.")


    def load_image(self):
        # Load and display the current image using Pillow
        pickle_file = self.pickle_files[self.current_image_index]
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
                white = np.log10(1 + np.abs(data['images']['white'])) #* self.mask
                white -= white.min()
                white /= white.max()
        except:
            white = np.zeros([200,200])

        self.current_angle = self.selections[os.path.basename(pickle_file)]
        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, str(self.current_angle))

        overlay = self.PSF_default[cropper(self.PSF_default, white.shape[-1])]
        rotated_image = rotate(white, -self.current_angle, reshape=False) * overlay
        rotated_image = Image.fromarray((rotated_image * 255).astype(np.uint8))
        rotated_image = rotated_image.resize((800, 800))

        self.current_image = ImageTk.PhotoImage(rotated_image)

        self.image_label.config(image=self.current_image)
        self.image_label.image = self.current_image  # Keep a reference!


    def change_angle(self, delta):
        self.current_angle = (self.current_angle + delta) % 360
        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, str(self.current_angle))
        self.apply_rotation()
        self.save_current_angle()  # Save the current angle to selections


    def set_angle(self):
        try:
            self.current_angle = int(self.angle_entry.get())
            self.apply_rotation()
            self.save_current_angle()  # Save the current angle to selections
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for the rotation angle.")


    def save_current_angle(self):
        current_image_name = os.path.basename(self.pickle_files[self.current_image_index])
        self.selections[current_image_name] = self.current_angle


    def apply_rotation(self):
        pickle_file = self.pickle_files[self.current_image_index]
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
                white = np.log10(1 + np.abs(data['images']['white'])) #* self.mask
                white -= white.min()
                white /= white.max()
        except:
            white = np.zeros([200,200])

        overlay = self.PSF_default[cropper(self.PSF_default, white.shape[-1])]
        rotated_image = rotate(white, -self.current_angle, reshape=False) * overlay
        rotated_image = Image.fromarray((rotated_image * 255).astype(np.uint8))
        rotated_image = rotated_image.resize((800, 800))

        self.current_image = ImageTk.PhotoImage(rotated_image)

        self.image_label.config(image=self.current_image)
        self.image_label.image = self.current_image  # Keep a reference!


    def save_labels(self, selections=None):
        if selections is None:
            selections = self.selections
        with open(self.labels_file, 'w') as f:
            for img, angle in selections.items():
                f.write(f"{img}: {angle}\n")


    def load_labels(self):
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(': ')
                    if len(parts) == 2:
                        img, angle_str = parts
                        self.selections[img] = float(angle_str)


    def next_image(self):
        self.save_labels()
        self.save_current_angle()  # Save the current angle before moving to the next image
        self.current_image_index = (self.current_image_index + 1) % len(self.pickle_files)
        self.load_image()


    def prev_image(self):
        self.save_labels()
        self.save_current_angle()  # Save the current angle before moving to the previous image
        self.current_image_index = (self.current_image_index - 1) % len(self.pickle_files)
        self.load_image()


# Example usage
if __name__ == "__main__":
    root = tk.Tk()

    default_font = font.nametofont("TkDefaultFont")
    default_font.configure(family="Helvetica", size=13)

    reduced_folder = MUSE_DATA_FOLDER + 'DATA_reduced/'
    pickle_files = [os.path.join(reduced_folder, f) for f in os.listdir(reduced_folder) if f.endswith('.pickle')]
    labels_file = os.path.join(MUSE_DATA_FOLDER, 'angles.txt')
    app = ImageLabeller(root, pickle_files, labels_file)
    root.mainloop()
