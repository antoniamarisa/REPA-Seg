from pathlib import Path 
import torch 
import argparse 

from scripts.training.trainer import Trainer

def args_parse():
    parser = argparse.ArgumentParser(description = "Segmentation baseline: Dinov2 backbone + ADE20K")
    
    # PATH
    parser.add_argument("--data-root", type = Path, default = "data/ADEChallengeData2016")
    parser.add_argument("--runs-root", type = Path, default = "runs")
    parser.add_argument("--exp-name", type = str, default = "repa-seg")

    # COMPUTATIONAL
    parser.add_argument("--device", type=str, default= "cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type = int, default = 18)
    parser.add_argument("--num-workers", type = int, default = 4)

    # PREPROCESSING
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--color-jitter", action="store_true")
    parser.add_argument("--augmentation-strategy", default="baseline", choices = ["baseline"])

    # TRAINING
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--epochs", type = int, default = 120)
    parser.add_argument("--batch-size", type = int, default = 16)
    parser.add_argument("--lr-backbone", type = float, default = 1e-4)
    parser.add_argument("--lr-head", type = float, default = 1e-3)
    parser.add_argument("--weight-decay", type = float, default = 0.05)
    parser.add_argument("--scheduler", type=str, default="poly_warmup", choices=[ "poly_warmup", "constant_warmup"])
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-ckpt-every", type=int, default=None, help="Save an additional checkpoint every N epochs, for later analysis (e.g. representation probing/similarity comparison). None disables intermediate saving.")

    # MODEL
    parser.add_argument("--backbone-model", type = str, default = "scratch/vit-small-14")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--img-size", type = int, default = 448)
    parser.add_argument("--num-classes", type = int, default = 150)
    parser.add_argument("--head-type", type=str, default = "mlp", choices = ["linear", "mlp"]) # extend if required

    # SCRATCH BACKBONE
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--att-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.0)

    # REPA
    parser.add_argument("--use-repa", action="store_true")
    parser.add_argument("--repa-proj", type= str, default= "3l-mlp-silu", choices= ["linear",  "3l-mlp-silu"]) # 3l-mlp-silu equals yu et al.
    parser.add_argument("--repa-lambda", type=float, default=0.5) # weight of repa loss relative to CE
    parser.add_argument("--repa-teacher", type=str, default="facebook/dinov2-base")
    parser.add_argument("--repa-teacher-layer", type=int, default = -1)   # choice of teacher layer for the representations that are used for REPA, -1 defaults to last layer
    parser.add_argument("--normalized-last-teacher-layer", action="store_true")
    parser.add_argument("--normalized-last-student-layer", action="store_true")
    parser.add_argument("--repa-student-layer", type=int, default = 4) # choice of student layer to align; midlayer fpr the ViT base would be 6 (as 12 layers in total)
    parser.add_argument("--lr-repa", type=float, default = 1e-4)
    # REPA Annealing motivated by Wang et al., 2025 
    # if both point to the same epoch, a hard-cut off is implemented
    parser.add_argument("--repa-decay-start", type = int, default=None) # epoch where deacy begins; None means no annealing
    parser.add_argument("--repa-decay-end", type = int, default = None) # epoch where lambda hits 0

    args = parser.parse_args()
    return args 

def main():
    args = args_parse()
    trainer = Trainer(args)
    trainer.train_loop()

if __name__ == "__main__":
    main()