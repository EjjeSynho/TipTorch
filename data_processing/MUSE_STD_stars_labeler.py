
#%%
import sys
sys.path.insert(0, '..')

import tkinter as tk
from tkinter import messagebox, ttk, font
import os
from PIL import Image, ImageTk

from MUSE_data_utils import MUSE_DATA_FOLDER

STD_FOLDER = MUSE_DATA_FOLDER / 'standart_stars/'

#%%
# from MUSE_STD_dataset_utils import update_label_IDs
# update_label_IDs()

#%%
class ImageLabeller:
    def __init__(self, master, image_paths, labels_file):
        self.master = master
        self.image_paths = image_paths
        self.labels_file = labels_file
        self.current_image_index = 0

        # Read list of classes from a file
        with open(os.path.join(STD_FOLDER, 'PSF_classes.txt'), 'r') as f:
            self.classes = [line.strip() for line in f]
        
        self.selections = {}

        # Store the current PhotoImage object
        self.current_image = None
        self.setup_gui()


    def setup_gui(self):
        self.master.title("Dataset Labeller")

        # Image display setup
        self.image_label = ttk.Label(self.master)
        self.image_label.pack()

        # Frame for the checkboxes
        self.check_frame = ttk.Frame(self.master)
        self.check_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Checkboxes for class selection
        self.check_vars = {}
        for i, cls in enumerate(self.classes):
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(self.check_frame, text=f"{i+1} - {cls}", variable=var, command=self.update_labels)
            chk.pack(anchor=tk.W, padx=30, pady=10)
            self.check_vars[cls] = var

        # Navigation buttons
        prev_button = ttk.Button(self.master, text="Previous", command=self.prev_image)
        prev_button.pack(side=tk.LEFT)
        next_button = ttk.Button(self.master, text="Next", command=self.next_image)
        next_button.pack(side=tk.RIGHT)

        # Bind arrow keys for navigation
        self.master.bind('<Left>',  lambda e: self.prev_image())
        self.master.bind('<Right>', lambda e: self.next_image())

        # Bind number keys to toggle checkboxes
        for i, cls in enumerate(self.classes, start=1):
            if i <= 9:  # Limit to the first 9 classes for simplicity
                self.master.bind(str(i), lambda event, idx=i-1: self.toggle_class(idx)(event))

        # Bind the "0" key to uncheck all checkboxes
        self.master.bind('0', self.clear_checkboxes)

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

        self.load_image()
        self.load_labels()


    def sort_selections_by_image_id(self):
        # Extract keys and sort them based on the image ID
        sorted_keys = sorted(self.selections.keys(), key=lambda x: int(x.split('_')[0]))

        # Create a new dictionary with sorted keys
        sorted_selections = {key: self.selections[key] for key in sorted_keys}

        # Update self.selections with the sorted dictionary
        self.selections = sorted_selections


    def jump_to_sample(self):
        # Get the entered sample ID from the textbox
        sample_id = self.id_entry.get()

        # Find the index of the image that matches the sample ID
        for index, path in enumerate(self.image_paths):
            # Assuming the sample ID is part of the filename, adjust as necessary
            if sample_id == os.path.basename(path).split('_')[0]:
                self.current_image_index = index
                self.load_image()
                break
        else:
            messagebox.showinfo("Info", f"Sample ID {sample_id} not found.")
            

    def toggle_class(self, index):
        """Generate a function to toggle the checkbox at the given index."""
        def toggle(event=None):  # Accept an optional event argument
            var = list(self.check_vars.values())[index]
            var.set(not var.get())
            self.update_labels()  # Update labels based on the new checkbox state
        return toggle


    def load_image(self):
        # Load and display the current image using Pillow
        image_path = self.image_paths[self.current_image_index]
        pil_image = Image.open(image_path)
        # pil_image = pil_image.resize((500, 500), Image.ANTIALIAS)  # Optional: resize image for display
        self.current_image = ImageTk.PhotoImage(pil_image)

        self.image_label.config(image=self.current_image)
        self.image_label.image = self.current_image  # Keep a reference!
        
        self.update_checkboxes()


    def clear_checkboxes(self, event=None):
        for var in self.check_vars.values():
            var.set(False)
        self.update_labels()  # Optionally update labels after unchecking all boxes


    def update_labels(self):
        # Update label file whenever a checkbox is checked/unchecked
        current_image = os.path.basename(self.image_paths[self.current_image_index])
        self.selections[current_image] = [cls for cls, var in self.check_vars.items() if var.get()]
        # self.sort_selections_by_image_id()
        self.save_labels()


    def update_checkboxes(self):
        # Get the current image's name
        current_image_name = os.path.basename(self.image_paths[self.current_image_index])
        # Retrieve the list of selected classes for the current image, if any
        selected_classes = self.selections.get(current_image_name, [])
        # self.sort_selections_by_image_id()
        # Iterate over the checkboxes and update their state
        for cls, var in self.check_vars.items():
            var.set(cls in selected_classes)


    def save_labels(self):
        # Save the current selections to the label file
        self.sort_selections_by_image_id()
        with open(self.labels_file, 'w') as f:
            for img, classes in self.selections.items():
                f.write(f"{img}: {', '.join(classes)}\n")


    def load_labels(self):
        # Load labels from the file if it exists
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(': ')
                    if len(parts) == 2:
                        img, classes_str = parts
                        self.selections[img] = classes_str.split(', ')
        self.update_checkboxes()


    def next_image(self):
        # Move to the next image
        self.save_labels()
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
        self.load_image()
        self.update_checkboxes()


    def prev_image(self):
        # Move to the previous image
        self.save_labels()
        self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
        self.load_image()
        self.update_checkboxes()


# Example usage
if __name__ == "__main__":
    root = tk.Tk()

    default_font = font.nametofont("TkDefaultFont")
    default_font.configure(family="Helvetica", size=13)

    image_folder = STD_FOLDER / 'MUSE_images'
    image_paths = [image_folder / f for f in os.listdir(image_folder) if f.endswith(('.png'))]
    # Sort by image IDs
    image_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

    labels_file = os.path.join(STD_FOLDER, 'labels.txt')
    app = ImageLabeller(root, image_paths, labels_file)
    root.mainloop()
