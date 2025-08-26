from torch.utils.data import Dataset
from PIL import Image
import os
import cv2

class HumanDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_transform=None, mask_transform=None):
        self.image_paths = sorted(os.path.join(images_dir, file) for file in os.listdir(images_dir))
        self.mask_paths = sorted(os.path.join(masks_dir, file) for file in os.listdir(masks_dir))
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)
        
        if self.img_transform:
            img = self.img_transform(img)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return img, mask
        
        