"""
Utilities for Qualitative analysis. 
Adjust onfig variables here if necessary. 
"""

from pathlib import Path
import numpy as np
import json
import torch
import torch.nn.functional as F
from models.scratch_vit_14_backbone import ScratchViT14Segm
import scripts.training.sliding_window as sw
import os
from dotenv import load_dotenv

# CONFIGS
TC_BASELINE = "tc202"
TC_REPA = "tc203"

BASELINE_EXP = "tc202_nr_final_ep200"
REPA_EXP = "tc205_r1_sl2_ablMAE_ep200"
# =====

load_dotenv()
#Set roots
REPO_ROOT = Path(os.environ["REPO_ROOT"])
DATA_ROOT = Path(os.environ["DATA_ROOT"])
RUNS_ROOT = Path(os.environ["RUNS_ROOT"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_ROOT = Path(f"{REPO_ROOT}/analysis/qualitative/{TC_BASELINE}_{TC_REPA}")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
SCORES_PATH = OUT_ROOT / f"per_image_scores_{TC_BASELINE}_{TC_REPA}.csv"

def get_best_checkpoint(exp_name):
    exp_dirs = []
    checkpoints = []
    for pth in RUNS_ROOT.rglob(exp_name):
        if pth.is_dir() and pth.name == exp_name:
            exp_dirs.append(pth)

    for dir in exp_dirs:
        for ckpt in dir.rglob("best.pth"):
            checkpoints.append(ckpt)
    assert(len(checkpoints) == 1)
    run_dir = checkpoints[0].parent.parent
    with open(run_dir/"config.json") as file:
        config = json.load(file)["config"]

    return {
        "checkpoint": checkpoints[0], "run_dir": run_dir, "config": config,
    }

def load_model_weights(model_info):
    config = model_info["config"]
    model = ScratchViT14Segm(return_features=False, img_size=448, patch_size=14,
            mlp_ratio=config["mlp_ratio"], att_dropout=config["att_dropout"], dropout=config["dropout"], 
            seg_head_type=config["head_type"], drop_path=config["drop_path"])
    checkpoint = torch.load(model_info["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict = True)
    model.to(DEVICE).eval()
    print("Loading complete...")
    return model


@torch.inference_mode()
def predict_sample_images(model, sample, img_size, num_classes, patch_size ):
    pixel_values = sample["image"].unsqueeze(0).to(DEVICE)
    mask = sample["mask"]

    logits_windowed =sw.sliding_window_predict(pixel_values, window_size=img_size, 
                        window_stride=img_size-32, window_batch_size=4,device=DEVICE.type, num_classes=num_classes, 
                       model=model, patch_size=patch_size)
    logits = F.interpolate( # resize to mask resolution
        logits_windowed.unsqueeze(0), size=mask.shape[-2:],
        mode="bilinear", align_corners=False,
    )
    # Get predicted class per pixel   
    predicted = logits.argmax(dim=1)

    return predicted[0].cpu().numpy().astype(np.int32)