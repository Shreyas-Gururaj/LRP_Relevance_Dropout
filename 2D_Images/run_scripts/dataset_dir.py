import torch
from torchvision import datasets
import os 

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        filename = os.path.split(path)[-1]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (filename,))

        return tuple_with_path
    