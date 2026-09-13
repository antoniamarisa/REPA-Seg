
"""
This script renders samples of gt_masks, pred masks and images for compared repa and no repa runs to find cases
of success and of failure for repa
"""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from preprocessing.transforms import album_transforms
from preprocessing.ade20k_dataset import ADE20KSegm
import scripts.qualitative_analysis.qualitative_utils as qs


SEED = 18
MIN_VALID_FRACTION = 0.50 # to dont have images with many background pixels
N_CANDIDATES = 100
N_SELECT = 10

SAMPLES_PATH = qs.OUT_ROOT / f"selected_samples.csv"


def color_map(N=256):
    """
    This method is based on:
    https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    with slight adaptations. 
    """
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype =  'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])
    return cmap

def select_examples(valset):
    scores = pd.read_csv(qs.SCORES_PATH)

    # Only consider examples where BOTH models produce reasonably interpretable segmentations
    good_performing_samples = scores[(scores["baseline_miou"] >= 0.40) &(scores["repa_miou"] >= 0.40)].copy()

    success = good_performing_samples[good_performing_samples["delta_miou"] >= 0.10].copy()
    failure = good_performing_samples[good_performing_samples["delta_miou"] <= -0.10].copy()

    print(f"Images where both models reach mIoU >= 0.40: {len(good_performing_samples)}")
    print(f"REPA success candidates (Δ >= +0.10): {len(success)}")
    print(f"REPA failure candidates (Δ <= -0.10): {len(failure)}")

    # Random selection 
    n_success = min(N_SELECT, len(success))
    n_failure = min(N_SELECT, len(failure))

    success = success.sample(n=n_success, random_state=SEED).copy()
    failure = failure.sample(n=n_failure, random_state=SEED + 1).copy()

    success["case"] = "success"
    failure["case"] = "failure"

    selected = pd.concat([success, failure], ignore_index=True)
    selected.to_csv(SAMPLES_PATH, index=False)

    return selected


def save_overview(examples, case):
    """
    renders overvoew of sample cases
    """
    fig, axes = plt.subplots(len(examples), 4, figsize=(12, 2.15 * len(examples)))
    titles = ["Input Image", "Ground Truth", "No REPA", "REPA-Seg"]

    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold")
    for i, ex in enumerate(examples):
        panels = [ex["input"], ex["g_truth"], ex["baseline"], ex["repa"]]
        for j, panel in enumerate(panels):
            axes[i, j].imshow(panel, interpolation="nearest")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

        axes[i, 0].set_ylabel(f"{ex['image']}\nDelta mIoU = {ex['delta_miou']:+.3f}\nvalid = {ex['valid_fraction']:.0%}", fontsize=8)
    plt.tight_layout()
    fig.savefig(qs.OUT_ROOT / f"overview_{case}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def colorize(mask, palette):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < 150)
    rgb[valid] = palette[mask[valid]]
    return rgb

def colorize_prediction(pred, mask, palette):
    rgb = colorize(pred, palette=palette)
    rgb[mask == 255] = 0
    return rgb

def main():
    baseline_info = qs.get_best_checkpoint(qs.BASELINE_EXP)
    repa_info = qs.get_best_checkpoint(qs.REPA_EXP)
    baseline_config, repa_config = baseline_info["config"], repa_info["config"]

    _, val_transform = album_transforms(
        img_size=baseline_config["img_size"], patch_size=baseline_config["patch_size"], 
        color_jitter=baseline_config.get("color_jitter", False), horizontal_flip=baseline_config.get("horizontal_flip", False), 
        augmentation_strategy=baseline_config.get("augmentation_strategy", "baseline"), backbone=baseline_config["backbone_model"])
    valset = ADE20KSegm(root=qs.DATA_ROOT, mode="validation", transform=val_transform)

    selected = select_examples(valset)

    baseline_model = qs.load_model_weights(baseline_info)
    repa_model = qs.load_model_weights(repa_info)

    examples = []

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Rendering selected examples"):
        idx = int(row["index"])
        sample = valset[idx]
        gt_mask = sample["mask"].numpy().astype(np.int64)

        baseline_pred_mask = qs.predict_sample_images(baseline_model, sample=sample, img_size=baseline_config["img_size"],
                                        num_classes=150, patch_size=baseline_config["patch_size"])
        repa_pred_mask = qs.predict_sample_images(repa_model, sample, img_size=repa_config["img_size"],
                                        num_classes=150, patch_size=repa_config["patch_size"])

        image_path = valset.imgs_path / valset.imgs[idx]
        original = np.asarray(Image.open(image_path).convert("RGB"))

        palette = color_map(256)[1:151]
        ground_truth_rgb = colorize(gt_mask, palette)
        baseline_rgb = colorize_prediction(baseline_pred_mask, gt_mask, palette=palette)
        repa_rgb = colorize_prediction(repa_pred_mask, gt_mask, palette=palette)

        image_folder = qs.OUT_ROOT / row["case"] / Path(row["image"]).stem
        image_folder.mkdir(parents=True, exist_ok=True)

        Image.fromarray(original).save(image_folder / "input.png")
        Image.fromarray(ground_truth_rgb).save(image_folder / "ground_truth.png")
        Image.fromarray(baseline_rgb).save(image_folder / "baseline.png")
        Image.fromarray(repa_rgb).save(image_folder / "repa.png")

        examples.append({
            "case": row["case"], "image": row["image"], 
            "input": original, "g_truth": ground_truth_rgb, 
            "baseline": baseline_rgb, "repa": repa_rgb, 
            "delta_miou": row["delta_miou"], "valid_fraction": row["valid_fraction"]
        })

    success_examples = [ex for ex in examples if ex["case"] == "success"]
    failure_examples = [ex for ex in examples if ex["case"] == "failure"]

    save_overview(success_examples, "success")
    save_overview(failure_examples, "failure")

if __name__ == "__main__":
    main()