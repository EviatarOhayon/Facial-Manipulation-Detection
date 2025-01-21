"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        if index < len(self.real_image_names):
            # The index belongs to the real image dataset
            image_path = os.path.join(self.root_path, 'real', self.real_image_names[index])
            label = 0
        else:
            # The index belongs to the fake image dataset
            fake_index = index - len(self.real_image_names)
            image_path = os.path.join(self.root_path, 'fake', self.fake_image_names[fake_index])
            label = 1

        # Attempt to open the image file
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, label

       

    def __len__(self):
        """Return the number of images in the dataset."""
        #print('real_image length = ' , len(self.real_image_names))
        #print('fake_image length = ', len(self.fake_image_names))
        return  len(self.fake_image_names) + len(self.real_image_names)
