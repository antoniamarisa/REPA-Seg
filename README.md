# Representation Alignment for Semantic Segmentation (REPA-Seg)

This repository contains the code used for the bachelor's thesis:  
**Representation Alignment for Semantic Segmentation using a Vision Transformer**


[Antonia Marisa Härle, LMU München]

---

**Summary**:  
This thesis investigates Representation Alignment (REPA, [[1]](#ref-1)) as an auxiliary training objective for semantic segmentation using a Vision Transformer (ViT). The proposed approach, Representation Alignment for Segmentation (REPA-Seg), aims to enhance semantic segmentation training by aligning the representations of a student ViT with frozen, pretrained self-supervised representations.
REPA-Seg successfully increased similarity between student and teacher representations and improved segmentation training and performance for a small ViT trained from random initialization.
However, this work did not identify a benefit of REPA for training a supervised pretrained ViT in semantic segmentation. 

---


## Repository structure

```text
REPA-Seg/
├── analysis/           # Results of analysis (CKNNA, qualitative examples, mIoU curve)
├── models/             # Student model classes
├── notebooks/          # Result evaluation and visualization
├── preprocessing/      # ADE20K dataset loading and preprocessing
├── results/            # Configs and metrics files for the conducted experiments
├── scripts/            # Training, evaluation, and analysis scripts
├── slurm/
│   └── jobs/           # Slurm job scripts for the conducted experiments
├── .env.example        # template for machine specific paths
├── README.md
└── requirements.txt
```

---

## 1. Environment setup

Create and activate virtual envornment for the repo: 
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the envvironment template:
```bash
cp .env.example .env
```

Change machine-specific paths for your setup in [`.env`](.env):
```text
export REPO_ROOT=/path/to/REPA-Seg                 # where this repo lives on your machine
export DATA_ROOT=/path/to/ADEChallengeData2016     # where the dataset lives on your machine
export RUNS_ROOT=/path/to/runs                     # where you want your run results to live
```

Active `.env` after editing and before submitting jobs:
```bash
source .env
```


--- 

## 2. Dataset Setup

Experiments were conducted with the [ADE20K] dataset, which can be downloaded from the website ([ADE20K], [[2]](#ref-2)). 

The expected structure for training and validation images and annotations is: 

```text
$DATA_ROOT/
├── images/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

## 3. Training

The provided Slurm scripts reproduce the cluster configuration used for this thesis. Partition, QoS, resource specifications, and filesystem paths might need to be adapted for other systems.


### 3.1. Submit training job

Training can be performed using the slurm scripts. Please adapt them, if necessary. 

To perform the main comparison of the baseline scratch ViT-S/14 and the corresponding REPA-Seg experiment, you can use these commands:

**Baseline (tc202)**: [Baseline training config tc202](slurm/jobs/runs/tc202_nr_final_ep200.sh)
```bash
sbatch slurm/jobs/runs/tc202_nr_final_ep200.sh
```

**REPA-Seg experiment (tc203)**: [REPA-Seg training config tc203](slurm/jobs/runs/tc203_r1_sl2_final_ep200.sh) 
```bash
sbatch slurm/jobs/runs/tc203_r1_sl2_final_ep200.sh
```

If you only want to do test runs (use subset of ADE20K, 8 training epochs, save every 2 checkpoints)
- Uses flags: `--test-run \` , (`--epochs 8 \`) and c(`--save-ckpt-every 2 \`)
```bash
sbatch slurm/jobs/test_runs/test_tc202_nr_final_ep200.sh
sbatch slurm/jobs/test_runs/test_tc203_r1_sl2_final_ep200.sh
```

---

### 3.2. Experiment table of this thesis

Table of the experiments conducted in this thesis.  
Slurm scripts for execution are under `slurm/jobs/`.  
Saved `config.json` and `metrics.json` file are under `results/`.

| Test case | Student | Teacher | REPA | Lambda | Student layer | Epochs |
|---|---|---|---:|---|---|---|
| tc20 | Scratch ViT-S/14 | -- | No | -- | -- | 120 |
| tc21 | Scratch ViT-S/14 | DINOv2-B/14 | Yes | 0.5 | 6 | 120 |
| tc22 | Scratch ViT-S/14 | DINOv2-B/14 | Yes | 0.5 | 12 | 120 |
| tc23 | Scratch ViT-S/14 | DINOv2-B/14 | Yes | 0.5 | 4 | 120 |
| tc24 | Scratch ViT-S/14 | DINOv2-B/14 | Yes | 0.5 | 9 | 120 |
| tc25 | Scratch ViT-S/14 | DINOv2-B/14 | Yes | 0.5 | 2 | 120 |
| tc26 | Scratch ViT-S/14 | DINOv2-B/14 | Yes | 0.25 | 4 | 120 |
| tc27 | Scratch ViT-S/14 | DINOv2-B/14 | Yes | 0.75 | 4 | 120 |
| tc28 | Scratch ViT-S/14 | DINOv2-B/14 | Yes | 1.0 | 4 | 120 |
| tc29 | Scratch ViT-S/14 | DINOv2-B/14 | Yes | 0.25 | 2 | 120 |
| tc30 | Scratch ViT-S/14 | DINOv2-B/14 | Yes | 0.75 | 2 | 120 |
| tc31 | Scratch ViT-S/14 | DINOv2-B/14 | Yes | 1.0 | 2 | 120 |
| tc70 | Pretrained ViT-S/16 | -- | No | -- | -- | 80 |
| tc71 | Pretrained ViT-S/16 | DINOv2-B/14 | Yes | 1.0 | 2 | 80 |
| tc72 | Pretrained ViT-S/16 | DINOv2-B/14 | Yes | 0.5 | 6 | 80 |
| tc73 | Pretrained ViT-S/16 | DINOv2-B/14 | Yes | 0.5 | 2 | 80 |
| tc74 | Pretrained ViT-S/16 | DINOv2-B/14 | Yes | 0.5 | 9 | 80 |
| tc75 | Pretrained ViT-S/16 | DINOv2-B/14 | Yes | 0.25 | 6 | 80 |
| tc202 | Scratch ViT-S/14 | -- | No | -- | -- | 200 |
| tc203 | Scratch ViT-S/14 | DINOv2-B/14 | Yes | 1.0 | 2 | 200 |
| tc205 | Scratch ViT-S/14 | MAE-B/16 | Yes | 1.0 | 2 | 200 |


## 4. Experiment results
Each run creates its own run directory under the run folder, containing the following files: 

```text
<experiment-name>/
└── <job-id>_<timestamp>/
    ├── checkpoints/
    ├── logs/
    ├── config.json
    └── metrics.json
```

## 5. Analysis

This repository supports the analysis of the results.  
**Note**: Model checkpoints are not included in this repo becasue of their size. CKNNA and Qualitative analysis require model inference, and therefore require either retraining the corrsponding experiments or the already trained checkpoints. 

### 5.1. Semantic segmentation performance
An overview of observed semantic segmentation performance is given in the [Evaluation Notebook](notebooks/eval_results.ipynb).  
It provides:
1. Per-epoch mIoU comparison between baseline and REPA-Seg runs.
2. Computation of the epoch from which the mIoU difference remains above a specified threshold (+1 percentage point).
3. Summary tables, including LaTeX-formatted output.
4. Visualization of validation mIoU per epoch throughout training for the main experiments.

It can be reused for own runs and can be extended to additional runs if wanted.

---

### 5.2. Representation similarity analysis  

Representation similarity analysis using Centered Kernel Nearest-Neighbor Alignment (CKNNA, [[3]](#ref-3)).  
Compute CKNNA analysis for main comparison tc202 and tc203.  
Global Configs in the [CKNNA execution file](scripts/cknna_analysis/cknna_execute.py) need to be changed if CKNNA shall be computed for other tcs. 

1. Execute [CKNNA execution file](scripts/cknna_analysis/cknna_execute.py), e.g. with slurm script (execution on abaki)
```bash
sbatch slurm/jobs/analysis/cknna-analysis.sh
```
2. After CKNNA analysis computation set global configs in the [CKNNA generate plots file](scripts/cknna_analysis/cknna_plots.py) for the respective tcs. Then execute. 
```bash
python -m scripts.cknna_analysis.cknna_plots
```

Notes:
- Results will be saved in `/analysis/cknna`
- Only supported teacher: DINOv2-B/14

--- 

### 5.3. Qualitative examples 

Qualitative examples visualization.  
Global Configs in the [Qualitative Utils file](scripts/cknna_analysis/cknna_execute.py) need to be changed if qualitative examples shall be selected for other tcs. 

1. Compute image mIoU scores for the main comparison tc202 and tc203, e.g. with slurm script (execution on abaki)
```bash
sbatch slurm/jobs/analysis/score-repa.sh
```
2. After image mIoU computation for main comparison tc202 and tc203, render example cases, e.g. with slurm script
```bash
sbatch slurm/jobs/analysis/render-repa.sh
```

Notes:
- Results will be saved in `/analysis/qualitative`
- Only supported student is: scratch ViT-S/14

---

## Notes

The following content is excluded from this repo to keep it lightweight:
- ADE20K dataset
- pretrained model weights
- saved training checkpoints
- TensorBoard event files 
- generated caches and log files 

## References

<a id="ref-1"></a>
[1] S. Yu et al., "Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think," ICLR, 2025. [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/d9e42b4d7163931f3689d6d6fbaa11d0-Paper-Conference.pdf)

<a id="ref-2"></a>
[2] B. Zhou et al., "Scene Parsing through ADE20K Dataset", CVPR, 2017. [Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_Scene_Parsing_Through_CVPR_2017_paper.html)

<a id="ref-3"></a>
[3] M. Huh et al., "The platonic representation hypothesis", ICML, 2024. [Paper](https://proceedings.mlr.press/v235/huh24a.html)


[ADE20K]: https://groups.csail.mit.edu/vision/datasets/ADE20K/