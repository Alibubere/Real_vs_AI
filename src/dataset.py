import torch
import os
import logging
from PIL import Image
from torch.utils.data import Dataset


class Real_vs_AI(Dataset):

    def __init__(self,data_dir,transform=None):
        """
        Custom dataset for Real vs AI images.

        Args:
            data_dir (str): Path to the dataset directory.
                            Expected structure:
                            data_dir/
                                real/
                                    img1.jpg
                                    img2.jpg
                                ai/
                                    img3.jpg
                                    img4.jpg
            transform (callable, optional): Transformations to apply to images.
        """
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        label_map = {"AiArtData":1,"RealArt":0}

        for label_name , label in label_map:

            folder_path = os.path.join(data_dir,label_name)
            
            if not os.path.exists(folder_path):
                logging.error(f"folder path does not exist {folder_path}")
                raise  

            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.png','.jpeg','.jpg','.gif')):
                    self.image_paths.append((os.path.join(folder_path,file_name)))
                    self.labels.append(label)

    def __len__(self):
        """Return total number of samples."""
        return len(self.image_paths)
    

    def __getitem__(self, index):
        """Load and return a sample from the dataset."""
        try:
            img_path = self.image_paths[index]
            label = self.labels[index]

            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image , label
        
        except IndexError:
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")
        except Exception as e:
            raise RuntimeError(f"Error loading image at index {index}: {e}")