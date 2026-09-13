"""
This script computes the mIoU for each image for a repa and non-repa comparison.
1. loads both models
2. rebuilds validation pipeline similar to training validation (sliding window and augmentation): but comparison happens on the untransformed image
3. Loop to get sample images:
    - get ground truth mask, run sliding window inference
    - resize each models prediciton to native resolution 
    - compute per image miou 
    - append all rows and get csv with all images 
"""

import numpy as np  
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from preprocessing.transforms import album_transforms
from preprocessing.ade20k_dataset import ADE20KSegm
from sklearn.metrics import confusion_matrix

import scripts.qualitative_analysis.qualitative_utils as qs

def calc_image_miou(pred_mask, gr_truth_mask, num_classes):
    """
    calculates mIoU on one image level.
    """
    valid = gr_truth_mask != 255
    pred_mask, gr_truth_mask = pred_mask[valid], gr_truth_mask[valid]

    conf_matrix = confusion_matrix(gr_truth_mask, pred_mask, labels=np.arange(num_classes))
    intersection = np.diag(conf_matrix) # diag of cm is inersection
    union = conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - intersection
    present = union > 0 # only non zeros union classes included 
    return float(np.mean(intersection[present] / union[present]))


def main():
    baseline_info = qs.get_best_checkpoint(qs.BASELINE_EXP)
    repa_info = qs.get_best_checkpoint(qs.REPA_EXP)

    baseline_config= baseline_info["config"]
    repa_config= repa_info["config"]

    baseline_model = qs.load_model_weights(baseline_info)
    repa_model = qs.load_model_weights(repa_info)

    _, val_transform = album_transforms(
        img_size=baseline_config["img_size"], patch_size=baseline_config["patch_size"], 
        color_jitter=baseline_config.get("color_jitter", False), horizontal_flip=baseline_config.get("horizontal_flip", False), 
        augmentation_strategy=baseline_config.get("augmentation_strategy", "baseline"), backbone=baseline_config["backbone_model"])
    valset = ADE20KSegm(root=qs.DATA_ROOT , mode="validation", transform=val_transform)

    if qs.SCORES_PATH.exists():
        existing = pd.read_csv(qs.SCORES_PATH)
        completed = set(existing["image"])
    else:
        existing = pd.DataFrame()
        completed = set()

    rows = []
    for idx in tqdm(range(len(valset)), desc="Scoring validation set"):
        image_name = valset.imgs[idx]
        if image_name in completed:
            continue

        sample = valset[idx]
        gr_truth_mask = sample["mask"].numpy().astype(np.int64)
        baseline_pred_mask = qs.predict_sample_images(baseline_model, sample=sample, img_size=baseline_config["img_size"],
                                        num_classes=150, patch_size=baseline_config["patch_size"])
        repa_pred_mask = qs.predict_sample_images(repa_model, sample, img_size=repa_config["img_size"],
                                        num_classes=150, patch_size=repa_config["patch_size"])

        # Get per images mIoUs
        baseline_miou = calc_image_miou(baseline_pred_mask, gr_truth_mask, baseline_config["num_classes"])
        repa_miou = calc_image_miou(repa_pred_mask, gr_truth_mask, baseline_config["num_classes"])
        valid = gr_truth_mask != 255
        valid_fraction = float(valid.mean())
        num_gr_truth_classes = len(np.unique(gr_truth_mask[valid]))

        rows.append({
            "index": idx, "image": image_name, "baseline_miou": baseline_miou, 
            "repa_miou": repa_miou, "delta_miou": repa_miou - baseline_miou, 
            "num_gr_truth_classes": num_gr_truth_classes, "valid_fraction": valid_fraction})
        current = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(subset="image", keep="last").sort_values("index")
        current.to_csv(qs.SCORES_PATH, index=False)

    df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(subset="image", keep="last").sort_values("index").reset_index(drop=True)
    df.to_csv(qs.SCORES_PATH, index=False)
    print(f"Finished. Saved {len(df)} rows to {qs.SCORES_PATH}")


if __name__ == "__main__":
    main()