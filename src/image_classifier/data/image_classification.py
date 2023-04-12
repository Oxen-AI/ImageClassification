
import pandas as pd
import numpy as np
from PIL import Image
import os
from enum import Enum
from torch.utils.data import Dataset


class ImageType(Enum):
    RGB = 1
    GRAYSCALE = 2

class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        annotations_file: str,
        img_size: tuple = (1, 28, 28),
        img_type: ImageType = ImageType.GRAYSCALE,
        load_data: bool = True,
    ):
        if len(img_size) != 3:
            print(f"Invalid img_size: {img_size}")
            raise ValueError('img_size must be a tuple of length 3')

        self.img_size = img_size
        self.img_type = img_type
        self.img_dir = img_dir

        # Might not want to load data if we are just using the load_image function
        if load_data:
            images, labels = self.load_from_csv(img_dir, annotations_file)
            self.images = images
            self.labels = labels
    
    # For iterating over the dataset
    def __len__(self):
        return len(self.labels)
    
    # For iterating over the dataset
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
    def load_from_csv(self, dir: str, path: str):
        df = pd.read_csv(path)
        return self.load_data_from_df(dir, df)
    
    def load_image_from_dir(self, dir: str, path: str):
        return self.load_image(os.path.join(dir, path))
    
    def load_image(self, full_path: str):
        # Load the image file using PIL
        image = Image.open(full_path)
        
        if self.img_type == ImageType.GRAYSCALE:
            # Convert the image to grayscale
            image = image.convert("L")

        # Resize the image to the proper size
        image = image.resize((self.img_size[1], self.img_size[2]))
        
        # Convert the image to a numpy array and store it in image_data
        return np.array(image).reshape(1, self.img_size[0], self.img_size[1], self.img_size[2]).astype(np.float32)
    
    def load_data_from_df(self, dir: str, df: pd.DataFrame):
        # Get the paths to the image files from the 'file' column
        image_paths = df['file'].tolist()

        # Create an empty numpy array to store the image data
        image_data = np.empty((len(image_paths), self.img_size[0], self.img_size[1], self.img_size[2]), dtype=np.float32)

        # Create an empty numpy array to store the labels
        labels = np.empty(len(image_paths), dtype=int)

        # Loop over each image path and label in the dataframe
        for i, (image_path, label) in enumerate(zip(image_paths, df['label'])):
            # Load image from path
            image_data[i] = self.load_image_from_dir(dir, image_path)
            # Store the label in the labels array
            labels[i] = label

        return image_data, labels