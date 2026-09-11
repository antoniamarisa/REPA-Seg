"""
Training and Validation Transformation using Albumentations Library. 
Implementation is based on the MMSegmentation Augmentation Pipeline and Segmenter (Strudel et al., 2021) with adaptations. 

References:
Segmenter:
    Strudel et al.,
    "Segmenter: Transformer for Semantic Segmentation",
    ICCV 2021.
    https://github.com/rstrudel/segmenter/tree/master/segm

Sliding-window inference:
    https://github.com/rstrudel/segmenter/blob/master/segm/model/utils.py

MMSegmentation ADE20K pipeline:
    https://github.com/open-mmlab/mmsegmentation/blob/main/configs/_base_/datasets/ade20k.py

"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2
import numpy as np
import math

# Mean and Std of Imagenet
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]

# For google/vit/augreg/jax the pretrained normalization should be used (Steiner et al., 2021)
MEAN_AUGREG = [0.5, 0.5, 0.5]
STD_AUGREG  = [0.5, 0.5, 0.5]

def get_normalization(backbone):
    if 'augreg' in backbone:
        return MEAN_AUGREG, STD_AUGREG
    else:
        return MEAN_IMAGENET, STD_IMAGENET

class ValidationTransforms:
    """
    Single-scale sliding-window validation (based on Segmenter)
    Difference: Preserves the image at its native resolution if both spatial dimensions are at least window_size 
    - if necessary, isotropically upscale the image such that its shorter side equals window_size
    - Keep the segmentation mask at its original resolution
    - Sliding-window inference is performed later in train.py
    """
    def __init__(self, window_size, patch_size, mean, std):
        self.window_size = window_size # is technically img_size
        self.patch_size = patch_size

        self.normalize = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ])

    def __call__(self, image, mask):
        """
        Callable, so can be called by e.g. .transform()
        """
        h, w = image.shape[:2] # image.shape =(height,width,channels)
        # Resize only if the shorter side is smaller than window_size (Segmenter's rule) — keep aspect ratio
        if min(h, w) < self.window_size:  # window_size = 448
            # image is smaller than training resolution — upscale so shorter side = 448
            if h < w:
                new_h, new_w = self.window_size, round(w * self.window_size / h) # round to keep aspect ratio
            else:
                new_h, new_w = round(h * self.window_size / w), self.window_size
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # else: image already >= 448 on its shorter side —> leave original resolution

        transformed = self.normalize(image=image)
        return {
            "image": transformed["image"],           # (3, H', W'), H',W' >= window_size
            "mask": torch.from_numpy(mask.copy()).long(),  # original resolution 
        }

class TrainingTransforms:
    def __init__(self, crop_size, patch_size, mean, std, 
                 color_jitter=True, horizontal_flip=True, augmentation_strategy= "baseline"):
        self.crop_size = crop_size
        self.patch_size = patch_size

        # Refernce scale mimics mmsegs of (2048, 518) to (1792, 448) of a 448 crop
        self.reference_scale = (4 * self.crop_size), self.crop_size
        self.ratio_range = (0.75, 2.0) # less aggressive to reduce padding 
        print(self.ratio_range)
        self.keep_ratio = True
        self.image_interpolation = cv2.INTER_LINEAR
        self.mask_interpolation = cv2.INTER_NEAREST

        # Padding pixels get approx filled with the imagenet rgb value:  returns mean_pad_fill ≈ (124, 116, 104)
        mean_pad_fill = tuple(
            int(round(mean_in_channel* 255))
            for mean_in_channel in mean
        )
        if augmentation_strategy == "baseline":
            train_list = [
                A.PadIfNeeded(
                    min_height=crop_size, min_width=crop_size,
                    position="top_left",
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=mean_pad_fill,
                    fill_mask=255, # ignore index filled in 
                    p=1.0,
                ),
                A.RandomCrop(
                    height=crop_size, width=crop_size,
                    p=1.0,
                ),
            ]
            if horizontal_flip:
                train_list.append(
                    A.HorizontalFlip(p=0.5)
                )
            # if color_jitter:
                train_list.append(
                    A.ColorJitter( # similar to photometricdistortion in mmsegmentation
                        brightness=0.125,
                        contrast=0.5,
                        saturation=0.5,
                        hue=0.05,
                        p=0.5,
                    )
                )
            train_list.extend([
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ])
            self.transforms = A.Compose(
                train_list,
                additional_targets={
                    "valid_mask": "mask"
                }
            ) 
        else:
            raise ValueError(f"Training augmentation strategy {augmentation_strategy} is not implemented.")

    def __call__(self,image, mask):
        """
        Callable, so can be called by e.g .transform()
        """
        valid_mask = np.ones(mask.shape, dtype=np.uint8)

        image, mask, valid_mask = self._reference_resize( image=image, mask=mask, valid_mask=valid_mask)

        transformed = self.transforms(
            image=image,
            mask=mask,
            valid_mask = valid_mask,
        )
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"].long()
        valid_pixel_mask = (transformed["valid_mask"]!= 255)

        # # DELETE assertions can be deleted, if is working
        # assert image_tensor.shape[-2:] == (self.crop_size, self.crop_size,)
        # assert mask_tensor.shape[-2:] == (self.crop_size, self.crop_size,)
        # assert transformed["image"].shape[-2:] == transformed["mask"].shape[-2:]

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            # binary mask that assigns 0 or 1 to indicate if pixel is invlaid(padded) or valid
            "valid_pixel_mask": valid_pixel_mask.bool()
        }

    def _reference_resize(self, image, mask, valid_mask):
        
        # Refernce scale mimics mmsegs of (2048, 518) to (1792, 448) of a 448 crop
        reference_scale_long_side, reference_scale_short_side = self.reference_scale

        ratio = np.random.uniform(*self.ratio_range) # a bit less aggressive than mmsegms 05, 2.0
        long_side = reference_scale_long_side * ratio 
        short_side =reference_scale_short_side * ratio 

        h, w = image.shape[:2]
        original_smaller_side = min(h,w)
        scale = min (long_side / max(h,w), short_side/min(h,w))

        new_h, new_w = max(1, round(h*scale)), max(1, round(w*scale))

        # RESIZE
        image = cv2.resize(image, (new_w, new_h), interpolation=self.image_interpolation)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=self.mask_interpolation)
        valid_mask = cv2.resize(valid_mask, (new_w, new_h), interpolation=self.mask_interpolation)

        return image, mask, valid_mask 


def album_transforms(img_size, patch_size, color_jitter, horizontal_flip, augmentation_strategy, backbone): # add image size? 
    resize_size = None
    crop_size = None
    if img_size is None or patch_size is None:
        raise ValueError(f"no default img_size or patch_size is defined.")
    else:
        resize_size = img_size
        crop_size = img_size

    assert resize_size is not None and crop_size is not None
    assert resize_size % patch_size == 0, f"Resize_size= {resize_size} must be divdable by {patch_size}"
    assert crop_size % patch_size == 0, f"crop_size = {crop_size} must be divdable by {patch_size}"

    strategy = "baseline" if augmentation_strategy is None else augmentation_strategy
    mean, std = get_normalization(backbone)

    train_transforms = TrainingTransforms(
        crop_size=crop_size , patch_size=patch_size,
        mean=mean, std=std,
        horizontal_flip=horizontal_flip, color_jitter=color_jitter, augmentation_strategy=strategy,
    )

    # Segmenter-style single-scale sliding-window validation
    val_transforms = ValidationTransforms(
        window_size=resize_size, patch_size=patch_size, mean=mean, std=std,
    )
    return train_transforms, val_transforms

