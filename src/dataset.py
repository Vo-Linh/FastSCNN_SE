import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision


class PandaSet(Dataset):
    def __init__(self, images_dir, label_dir, images_list, transforms = None, num_classes = 13):
        self.images_dir = images_dir
        self.label_dir = label_dir
        self.transforms = transforms

        # remove .npy from name images
        self.list_images = images_list
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.list_images)

    def _make_one_hot(self, image):

        h, w= image.shape
        one_hot_encoded_labels = np.zeros((self.num_classes, h, w), dtype=np.float32)
        for i in range(0, self.num_classes):
            one_hot_encoded_labels[i, image - 1 == i] = 1

        return one_hot_encoded_labels

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, f"{self.list_images[idx][:-4]}.npy")
        label_path = os.path.join(self.label_dir, f"{self.list_images[idx][:-4]}.png")

        images = np.load(image_path)
        label = Image.open(label_path)

        if self.transforms:
            images = self.transforms(images)

        label = self._make_one_hot(np.array(label))
        image = images.type('torch.FloatTensor')
        
        return images, label