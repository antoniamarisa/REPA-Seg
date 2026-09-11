"""
ADE20K Dataset class used in REPA-Seg.

ADE20K scene parsing dataset introduced by:
    Zhou et al., "Scene Parsing through ADE20K Dataset",CVPR 2017.
    https://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_Scene_Parsing_Through_CVPR_2017_paper.html

Dataset website:
    https://groups.csail.mit.edu/vision/datasets/ADE20K/

Expected structure:
    $DATA_ROOT/
    ├── images/
    │   ├── training/
    │   └── validation/
    └── annotations/
        ├── training/
        └── validation/
"""

from pathlib import Path 
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

class ADE20KSegm(Dataset):
    def __init__(self, root, mode, transform):
        self.root_dir = root 
        self.mode = mode # validation vs. training
        self.transform = transform

        self.imgs_path = self.root_dir/"images" /mode
        self.masks_path = self.root_dir/"annotations" / mode

        # Ensure image list is in deterministic order
        self.imgs = sorted([i for i in os.listdir(self.imgs_path) if i.endswith('.jpg')])
    
    def __len__(self):
        """
        Called when DataLoader is created to know the total dataset size.
        """
        return len(self.imgs)
    
    def preprocess_mask(self, mask):
        """
        Remapping Labels: 
        - 0 -> 255 (ignored by loss; unlabeled pixels get mapped to 255, because in train.py CrossEntropyLoss(ignore_index=255)),
        - 1-150-> 0-149 (the original 150 classes get shifted down by 1 to be indexed from 0 on)
        Reason:
            - ADE20K stores labels as 1-150 in PNG files, with 0 as "background"
            - PyTorchs CrossEntropyLoss expects class indices starting from 0
            - shifting is required: with 0 labeled pixels get mapped to 255
        """
        mask = mask.astype(np.int64)
        mask[mask==0] = 255 # the unlabeled gets ignored
        mask[mask!=255] -= 1 # 1-150 --> 0-149
        return mask

    def __getitem__(self, index):
        """
        Build one batch by calling the method multiple times. 
        Collects all the individual samples and stacks them togetehr into a batch tensor. 
        """
        img_name = Path(self.imgs[index]).stem
        img_jpg = self.imgs_path / f"{img_name}.jpg"
        mask_png = self.masks_path / f"{img_name}.png"

        image = np.array(Image.open(img_jpg).convert("RGB"))
        mask = np.array(Image.open(mask_png)) # leave 2D Label-Map
        mask = self.preprocess_mask(mask)
    
        # transform aplly - Transform classes are callable 
        sample = self.transform(image= image, mask=mask)
        result = {
            "image": sample["image"], 
            "mask": sample["mask"].long()
        }
        if "valid_pixel_mask" in sample:
            result["valid_pixel_mask"] = sample["valid_pixel_mask"]

        return result