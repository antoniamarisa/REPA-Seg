"""
CKNNA representation similarity analysis for REPA-Seg.
"""

# IMPORTS
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TRF
from transformers import Dinov2Model

from models.scratch_vit_14_backbone import ScratchViT14Segm
from models.pretrained_vit_16_backbone import TimmViT16_Segm
import scripts.cknna_analysis.cknna_formula as cknna

# ==========================================================
# CONFIGS — CHANGE THESE 
# Select testcases
BASELINE_TC = 202
REPA_TC = 203

DINOV2B_TEACHER = "facebook/dinov2-base" # Only one supported
TRAINING_MDOE = "scratch" 
STUDENT_MODEL = "scratch/vit-small-14" 

# TRAINING_MDOE = "pretrained"
# STUDENT_MODEL = "timm/vit_small_patch16_224.augreg_in21k_ft_in1k"

# =============================================================

load_dotenv()
#Set roots
REPO_ROOT = Path(os.environ["REPO_ROOT"])
DATA_ROOT = Path(os.environ["DATA_ROOT"])
RUNS_ROOT = Path(os.environ["RUNS_ROOT"])


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = Path(f"{DATA_ROOT}/images/validation")
OUTPUT_ROOT = Path(f"{REPO_ROOT}/analysis/cknna/{BASELINE_TC}_{REPA_TC}")


IMG_SIZE  = 448
K_NEIGHBORS = 10
SEED = 18
BATCH_SIZE = 8

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
AUGREG_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
AUGREG_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


# PREPROCESSING 
class StudentModel:
    def __init__(self, tc, name, training_mode, checkpoint, use_repa, student_layer, repa_lambda):
        self.tc = tc 
        self.name = name 
        self.training_mode = training_mode
        self.checkpoint = checkpoint
        self.use_repa = use_repa
        self.student_layer = student_layer
        self.repa_lambda = repa_lambda

    def derive_from_tc(tc):
        exp_dir, run_dir, config_file, checkpoint = find_run(tc)

        student_layer, repa_lambda = None, None
        use_repa = config_file.get("use_repa", False)
        if use_repa:
            student_layer = config_file["repa_student_layer"]
            repa_lambda = config_file["repa_lambda"]

        student_layer = 12 if student_layer == -1 else student_layer

        return StudentModel(tc=tc, name= f"tc{tc}", checkpoint= checkpoint,training_mode=TRAINING_MDOE,
                              use_repa=use_repa, student_layer=student_layer,repa_lambda= repa_lambda)


# HELPERS

def get_current_tc(path_name):
    match_obj = re.search(r"(?:^|_)tc(?P<tc>\d+)(?:_|$)", path_name) 
    if match_obj:
        return int(match_obj.group("tc"))
    else:
        return match_obj
    
def find_run(tc):
    exp_dir = None
    for pth in RUNS_ROOT.iterdir():
        if pth.is_dir() and get_current_tc(pth.name) == tc:
            exp_dir = pth

    if exp_dir is None:
        raise RuntimeError(f"no run directory found for this tc: {tc}")

    for run_dir in sorted (pth for pth in exp_dir.iterdir() if pth.is_dir()):
        config_pth = run_dir / "config.json"
        last_ckpt_pth = run_dir /"checkpoints" / "last.pth"
        if config_pth.exists() and last_ckpt_pth.exists():
            with open(config_pth) as file:
                config_file =json.load(file)
                config_file = config_file.get("config", config_file)
                return exp_dir, run_dir, config_file, last_ckpt_pth
        else:
            raise RuntimeError(f"run folder is not complete as expected for tc: {tc}")

def iterate_image_batches (image_paths, batch_size):
    """
    Generates batches of batch_size of images, yielding one batch at  a time.
    """
    for init in range(0, len(image_paths), batch_size):
        yield image_paths[init:init+batch_size]

def resize_crop(image_path, img_size):
    """
    Resize to shortest size and then senter crop to assure every model sees the same preprcoessed image.
    """
    image = Image.open(image_path).convert("RGB")
    transform_list = [
        transforms.Resize(size=img_size), 
        transforms.CenterCrop(size=img_size), 
        transforms.ToTensor()
    ]
    transform = transforms.Compose(transform_list)
    
    return transform(image)

def normalize_student_patches(student_model, images):
    # get a list of per-layer tensors (student_layers) with each (B, N_patches, D),
    if 'timm' in STUDENT_MODEL:
        # L2 normalize and get representations and colelct them per layer
        normalized_pixel_values = TRF.normalize(images.to(DEVICE), AUGREG_MEAN.to(DEVICE), AUGREG_STD.to(DEVICE))
        hidden_states = student_model._hidden_states(normalized_pixel_values)
        student_layers = [hidden[:, student_model.num_prefix_tokens:, :].float() for hidden in hidden_states]
    else: 
        # L2 normalize and get representations and colelct them per layer
        normalized_pixel_values = TRF.normalize(images.to(DEVICE), IMAGENET_MEAN.to(DEVICE), IMAGENET_STD.to(DEVICE))
        _, hidden_states = student_model.backbone(pixel_values=normalized_pixel_values, return_features=True)
        student_layers = [hidden[:, 1:, :].float() for hidden in hidden_states] 
    return student_layers  


def save_cknna_single_results(config_dir, cknna_score):
    config_dir.mkdir(parents=True, exist_ok=True)

    # save as csv 
    with open(config_dir / "single_cknna.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["student_layer", "cknna"])

        for layer, score in enumerate(cknna_score):
            writer.writerow([layer, f"{score:.8f}"])

    np.save(config_dir / "cknna_score.npy", cknna_score)

def save_cknna_comparison_results(comparison_dir, repa_name, baseline_name, repa_cknna_score, baseline_cknna_score, aligned_student_layer, repa_config):
    comparison_dir.mkdir(parents=True, exist_ok=True)

    cknna_delta = repa_cknna_score - baseline_cknna_score

    # save as csv 
    with open(comparison_dir / "cknna_delta_per_layer.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["student_layer", "cknna_delta"])

        for layer, delta_value in enumerate(cknna_delta):
            writer.writerow([layer, f"{delta_value:+.8f}"])

    np.save(comparison_dir / "cknna_delta_per_layer.npy", cknna_delta)

    with open(comparison_dir / "cknna_summary.txt", "w") as f:
        f.write(f"repa_run={repa_name}\n")
        f.write(f"baseline_run={baseline_name}\n")
        f.write(f"aligned_student_layer={aligned_student_layer}\n")
        f.write(f"lambda={repa_config.repa_lambda:.2f}\n")
        f.write(f"k={K_NEIGHBORS}\n")
        f.write(f"baseline_cknna={baseline_cknna_score[aligned_student_layer]:.8f}\n")
        f.write(f"repa_cknna={repa_cknna_score[aligned_student_layer]:.8f}\n")
        f.write(f"delta_cknna={cknna_delta[aligned_student_layer]:+.8f}\n")


# TRAJECTORY COMPUTATION OVER MATCHING CHECKPOINTS
@torch.inference_mode()
def cknna_single_sl( checkpoint, layer_i, image_paths, teacher_features):
    config = StudentModel(tc=-1, name="trajectory", training_mode=TRAINING_MDOE, checkpoint=checkpoint, student_layer=None, repa_lambda=None, use_repa=False)
    student_model = build_student_model(config)

    image_batches =[]
    for batch_path in iterate_image_batches(image_paths=image_paths, batch_size=BATCH_SIZE):
        original_images = (torch.stack([resize_crop(image, IMG_SIZE) for image in batch_path], dim=0))
        student_layers = normalize_student_patches(student_model, original_images) 

        image_batches.append(student_layers[layer_i].mean(dim=1).cpu())
        del student_layers, original_images

    image_batches = torch.cat(image_batches, dim=0).float()
    student_features = F.normalize(image_batches.to(DEVICE), dim=-1)
    teacher_image = teacher_features.to(DEVICE).float()
    score = cknna.cknna_formula(Ks=student_features, Lt= teacher_image, k_neighbors=K_NEIGHBORS, device=DEVICE)

    del student_model, student_features

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return score

def build_trajectory(repa_config, baseline_config, aligned_sl, image_paths, teacher_feature, comparison_directory):
    # FInd. matching checkpoints
    repa_checkpoints, baseline_checkpoints = [], []
    for path in sorted(repa_config.checkpoint.parent.glob("epoch_*.pth")):
        epoch = int(path.stem.split("_")[1])
        repa_checkpoints.append((epoch, path))

    for path in sorted(baseline_config.checkpoint.parent.glob("epoch_*.pth")):
        epoch = int(path.stem.split("_")[1])
        baseline_checkpoints.append((epoch, path))
    repa_checkpoints, baseline_checkpoints = dict(repa_checkpoints), dict(baseline_checkpoints)
    matched_epoch_pths = sorted(set(repa_checkpoints).intersection(baseline_checkpoints))

    if not matched_epoch_pths:
        print("no matched epochs found, so no trajectory is compued")
        return

    trajectory_csv_pth = comparison_directory / "trajectory.csv"
    comparison_rows = []

    # Iteratre over matching epochs ( should be every tenth)
    for epoch_i in matched_epoch_pths:
        epoch = epoch_i + 1 # since it starts at 0
        print(f"Trajectory: epoch {epoch}")

        baseline_cknna_score = cknna_single_sl(baseline_checkpoints[epoch_i], aligned_sl, image_paths, teacher_feature)
        repa_cknna_score = cknna_single_sl( repa_checkpoints[epoch_i], aligned_sl, image_paths, teacher_feature)

        comparison_rows.append({
            "epoch": epoch,
            "baseline_cknna_score": baseline_cknna_score,
            "repa_cknna_score": repa_cknna_score,
            "delta_cknna_score": repa_cknna_score - baseline_cknna_score,
        })

    # write csv
    with open(trajectory_csv_pth, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "baseline_cknna_score", "repa_cknna_score", "delta_cknna_score"])
        writer.writeheader()
        writer.writerows(comparison_rows)


# ETRACT THE FEATURE REPRESENTATIONS OF STUDENT AND TEACHER
def build_student_model(student_config):
    if STUDENT_MODEL == "scratch/vit-small-14":
        student_model = ScratchViT14Segm(img_size=IMG_SIZE, patch_size=14, embedding_dim=384, n_layers=12, n_heads=6, n_classes=150)
    elif 'timm' in STUDENT_MODEL:
        student_model = TimmViT16_Segm(backbone=STUDENT_MODEL, n_classes=150, return_features=True, head_type="linear", pretrained=True)
    else: 
        raise ValueError(f"Unsopported Student: {STUDENT_MODEL}")

    checkpoint= torch.load(student_config.checkpoint, map_location='cpu', weights_only=False)
    student_model.load_state_dict(checkpoint["model"])

    return student_model.to(DEVICE).eval()


# similar to toch.no_grad()
@torch.inference_mode()
def get_normalized_teacher_patches_per_batch(teacher, images):
    """
    Preprocessing of batch to L2 normalied pixel tensor
    """
    images = TRF.normalize(images.to(DEVICE), IMAGENET_MEAN.to(DEVICE), IMAGENET_STD.to(DEVICE))
    out = teacher(pixel_values=images, output_hidden_states=False, interpolate_pos_encoding= True, return_dict=True)
    return out.last_hidden_state [:,1:,:].float()

@torch.inference_mode()
def extract_teacher_features_per_image(teacher, image_paths):
    """
    Processes the images in batches, mean-pooles across the image.
    Returns all images L2-normalized and mean pooled per row across whole dataset
    """
    image_batches = []
    for batch_path in iterate_image_batches(image_paths=image_paths, batch_size=BATCH_SIZE):
        # load image batch
        original_images = (torch.stack([resize_crop(image, IMG_SIZE) for image in batch_path], dim=0))
        teacher_patches = get_normalized_teacher_patches_per_batch(teacher, original_images)
        # teacher_patches[i] are now the patch representations of one image
        # Mean pooling across patch dimesnion to get L2 image level mean 
        image_batches.append(teacher_patches.mean(dim=1).cpu())
        del teacher_patches, original_images

    image_batches = torch.cat(image_batches, dim=0).float()
    return F.normalize(image_batches, dim = -1)

# LAYER WISE CKNNA ANALYSIS
@torch.inference_mode()
def cknna_analysis(student_config, image_paths, teacher_image_features):
    """
    MAin method that performs the CKNNA analysis.
    - builds and loads student model (baseline or repa)
    - processes the images in batches: for each batch it extracts the patch token representations of the student per layer
    - mean pooles those patch tokens to one tensor per image and  builds a image tensor list per layer  
    
    - then builds the per layer CKNNA scores by accumulations all per batch vectors and L2 normalize them 
    - CKNNA is computed to the teacher reference layer table

    - returns an array of image-level CKNNA scores per student layer
    
    """
    student_model = build_student_model(student_config=student_config)

    layer_batches =  None 
    for batch_path in iterate_image_batches(image_paths=image_paths, batch_size=BATCH_SIZE):
        original_images = (torch.stack([resize_crop(image, IMG_SIZE) for image in batch_path], dim=0))
        student_layers = normalize_student_patches(student_model, original_images)

        if layer_batches is None:
            # fill empty list once for the very first batch
            layer_batches = [[] for _ in range(len(student_layers))]

        for layer_i, student in enumerate(student_layers):
            # mean poole across patch dimension for every per layer tensor in this batch
            layer_batches[layer_i].append(student.mean(dim=1).cpu())
            # get one layer tensor per image in this batch and put it in the list

        del student_layers, original_images # frree the tensors

    # Now compute CKNNA score per layer for the collected per-batch pooled vectors
    teacher_image = teacher_image_features.to(DEVICE).float()
    cknna_per_layer = np.zeros(len(layer_batches), dtype=np.float64)

    for layer_i, layer_batch in enumerate(layer_batches):
        # loop over the per batch pooles vectors for each layer
        layer_batch = torch.cat(layer_batch, dim=0).float()
        student_image = F.normalize(layer_batch.to(DEVICE), dim=-1) # L2-normalize

        # COMPUTER CKNNA SCORE
        cknna_per_layer[layer_i] = cknna.cknna_formula(Ks=student_image, Lt= teacher_image, k_neighbors=K_NEIGHBORS, device=DEVICE)

        del student_image

    del student_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cknna_per_layer

def main():
    image_paths = sorted(IMAGE_DIR.glob("*jpg"))

    repa_config = StudentModel.derive_from_tc(REPA_TC)
    baseline_config = StudentModel.derive_from_tc(BASELINE_TC)

    # BUILD SAVING DIR
    cknna_dir = f"tc{BASELINE_TC}_vs_tc{REPA_TC}_sl{repa_config.student_layer}_{repa_config.repa_lambda}"
    cknna_dir = OUTPUT_ROOT / cknna_dir
    single_dir = cknna_dir / "single"
    comparison_dir = cknna_dir / f"comparison_tc{BASELINE_TC}vs.tc{REPA_TC}"
    baseline_dir = single_dir /f"tc{BASELINE_TC}" 
    repa_dir = single_dir / f"tc{REPA_TC}"

    baseline_dir.mkdir(parents=True, exist_ok=True)
    repa_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    print(f"CKNNA analysis is running for tc{BASELINE_TC} and tc{REPA_TC}")
    print(f"\nLoading the teacher and pooling images...")

    # Write list of image dirse:
    teacher = Dinov2Model.from_pretrained(DINOV2B_TEACHER).to(DEVICE).eval()
    teacher_features = extract_teacher_features_per_image(teacher=teacher, image_paths=image_paths)

    print(f"\nAnalyzing tc{BASELINE_TC}...")
    baseline_cknna_score = cknna_analysis(baseline_config, image_paths, teacher_features)
    save_cknna_single_results(baseline_dir, baseline_cknna_score)

    print(f"\nAnalyzing tc{REPA_TC}...")
    repa_cknna_score = cknna_analysis(repa_config, image_paths, teacher_features)
    save_cknna_single_results(repa_dir, repa_cknna_score)

    save_cknna_comparison_results(
        comparison_dir=comparison_dir,
        repa_name = f"tc{REPA_TC}",
        baseline_name = f"tc{BASELINE_TC}",
        repa_cknna_score = repa_cknna_score,
        baseline_cknna_score=baseline_cknna_score,
        aligned_student_layer=repa_config.student_layer,
        repa_config=repa_config
    )

    build_trajectory(
        repa_config=repa_config,
        baseline_config=baseline_config,
        aligned_sl=repa_config.student_layer,
        image_paths=image_paths,
        teacher_feature=teacher_features,
        comparison_directory=comparison_dir,
    )

    del teacher

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"CKNNA analysis completed and saved to {cknna_dir}")

if __name__ == "__main__":
    main()
