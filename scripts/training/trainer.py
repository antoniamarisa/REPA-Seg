"""
Training Routine of student model: baseline and wired REPA-Seg branch (--use-repa)
"""

import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import evaluate
from tqdm.auto import tqdm
import random, numpy as np
import json
import os
import math
from transformers import AutoConfig
import torch.nn.functional as F
import timm 

from preprocessing.transforms import album_transforms
from preprocessing.ade20k_dataset import ADE20KSegm
from models.scratch_vit_14_backbone import ScratchViT14Segm
from models.pretrained_vit_16_backbone import TimmViT16_Segm
from scripts.training.repa_loss import REPALoss
import scripts.training.sliding_window as sw


def seed_worker(worker_id):
    """
    Makes seed worker specific, since otherwise there might be correletad randomness between worker.
    """
    worker_seed = torch.initial_seed() %2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer(object):
    def __init__(self, args):
        self.args = args 
        self.start_time = datetime.now()
        # set seed
        self._set_seed()

        self.device = torch.device(args.device)
        self.n_classes = self.args.num_classes

        # infer patch size and test if img_size is divisible by that
        self.patch_size = self._infer_patch_size()
        if self.args.img_size is not None and  self.args.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size={self.args.img_size} must be divisible by patch_size={self.patch_size}"
            )

        # INITIALIZE DATA 
        self._init_data()
       
        # STUDENT MODEL
        self.model = self._build_model()

        # REPA INSTANTIATION 
        self._init_repa()
        
        # OPTIMIZER
        self.optimizer = self._build_optimizer()
        # self.model = torch.compile(self.model) # For efficiency - not tested yet

        self.criterion = nn.CrossEntropyLoss(ignore_index=255) 

        self.run_dir, self.ckpt_dir, self.log_dir = self._make_run_dirs(self.args.exp_name)
        
        if self.args.resume is None:
            self._write_config()
        self.writer = SummaryWriter(str(self.log_dir))

        self.global_step = 0 # iteration counter for all training batches over all epochs
        self.best_miou = -1.0
        self.best_epoch = -1
        self.start_epoch = 0 

        # SCHEDULER
        self.scheduler = self._build_scheduler()

        # SCALER
        # (by default Pytorch uses 32-bit floats everywhere and with AMP (automatic mixed precision
        # pytorch uses 16-bit floats for most operations and only uses 32 when numerical precision is necessary))
        self.scaler = torch.amp.GradScaler( "cuda", enabled=(self.device.type=="cuda"))

        self.metrics_log = []

        # RESUME CHECKPOINT
        if self.args.resume is not None:
            self._init_resume_checkpoint()

    def _init_data(self):
        self.train_transforms, self.val_transforms = album_transforms(
            self.args.img_size, self.patch_size, self.args.color_jitter, 
            self.args.horizontal_flip, self.args.augmentation_strategy, self.args.backbone_model
        )

        self.trainset = ADE20KSegm(root= self.args.data_root, mode="training", transform=self.train_transforms)
        self.valset = ADE20KSegm(root = self.args.data_root, mode = "validation", transform = self.val_transforms)

        # TEST RUNS
        if self.args.test_run:
            self.trainset = Subset(self.trainset, range(64))
            self.valset = Subset(self.valset, range(64))

        # DATALOADERS
        # handle feeding data to the model during training (Batching, shuffling, parallel loading) (adding collate_fn custom function?)
        persistent_workers = self.args.num_workers > 0
        pin = (self.device.type == "cuda")
        # Make seeding reproducible
        self.dataloader_generator = torch.Generator()
        self.dataloader_generator.manual_seed(self.args.seed)
        self.train_loader = DataLoader(
            self.trainset, 
            batch_size = self.args.batch_size, 
            shuffle=True, 
            num_workers=self.args.num_workers, # num_workers controls how many parallel CPU processes load batches, while GPU is busy training
            pin_memory=pin, 
            persistent_workers= persistent_workers,
            worker_init_fn=seed_worker,
            generator = self.dataloader_generator,
        ) # True here if there are num_workers; persistent_workers controls whether worker processes are kept alive between epochs
        self.val_loader= DataLoader(self.valset, batch_size=1,# changed to 1 as the samples can have different shape  -> could also define new collate function instead
                                    shuffle = False, num_workers=self.args.num_workers, 
                                    pin_memory=pin, persistent_workers= persistent_workers, worker_init_fn=seed_worker)

    def _init_repa(self):
       # REPA Decay check
        if (self.args.repa_decay_start is None) != (self.args.repa_decay_end is None):
            raise ValueError("Arguments --repa-decay-start and --repa-decay-end must both be set or both be None.")
           
        # if use_repa == True, instantiate RepA Loss
        self.repa_loss = None
        if self.args.use_repa:
            if self.args.backbone_model.startswith("scratch/vit"):
                student_dim = self.model.backbone.embedding_dim
            elif self.args.backbone_model.startswith("timm/vit"):
                student_dim = self.model.backbone.embed_dim
            else:
                student_dim = self.model.backbone.config.hidden_size # student embedding size; extracts how wide each token embedding vector is 
            self.repa_loss = REPALoss(
                student_dim=student_dim, 
                teacher = self.args.repa_teacher,
                teacher_layer= self.args.repa_teacher_layer,
                normalized_last_teacher_layer=self.args.normalized_last_teacher_layer,
                repa_proj=self.args.repa_proj,
                student_backbone=self.args.backbone_model,
            ).to(self.device) # https://huggingface.co/docs/transformers/en/model_doc/dinov2

    def _init_resume_checkpoint(self):
        ckpt = torch.load(self.args.resume, map_location=self.device, weights_only=False)
        self._load_ckpt(ckpt)
        if "scaler" in self._resume_ckpt:
            self.scaler.load_state_dict(self._resume_ckpt["scaler"])

        # saving metrics.json
        path = self.run_dir / "metrics.json"
        with open(path) as f:
            self.metrics_log = json.load(f)["epochs"]
        self.metrics_log = [e for e in self.metrics_log if e["epoch"] < self.start_epoch]
    
    # PUBLIC METHODS
    def train_loop(self):

        # Training loop:
        for epoch in range(self.start_epoch, self.args.epochs):
            # TRAIN
            train_loss, ce_loss, repa_loss, repa_cosine = self._train_one_epoch(epoch)

            # VALIDATION
            val_loss, metrics = self._validation(epoch)

            # LOG
            self.writer.add_scalar("train/loss_per_epoch", train_loss, epoch)
            self.writer.add_scalar("train/ce_loss_per_epoch", ce_loss, epoch)
            if self.args.use_repa:
                self.writer.add_scalar("train/repa_loss_per_epoch", repa_loss, epoch)
                self.writer.add_scalar("train/repa_cosine_per_epoch",repa_cosine, epoch)
            self.writer.add_scalar("val/loss_per_epoch", val_loss, epoch)
            self.writer.add_scalar("val/mean_iou", metrics["mean_iou"], epoch)
            self.writer.add_scalar("val/mean_accuracy", metrics["mean_accuracy"], epoch)
            self.writer.add_scalar("val/overall_accuracy", metrics["overall_accuracy"], epoch)

            # Log LRs of all optimizer groups
            for i, group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f"train/lr_group_{i}", group["lr"], epoch)

            # Log per category iou as histogram:
            ctg_iou = metrics.get("per_category_iou", None)
            if ctg_iou is not None:
                ctg_iou_array = np.array(ctg_iou, dtype = np.float32)
                ctg_iou_array = ctg_iou_array[~np.isnan(ctg_iou_array)] # keep all non-NaN's (removes NaN's, as some classes might be not represented in the validation set)
                if ctg_iou_array.size > 0: # False if all values are nan
                    self.writer.add_histogram("val/per_category_iou_hist", ctg_iou_array, epoch)

            # CHECKPOINT
            miou = metrics["mean_iou"]
            is_best = miou > self.best_miou
            if is_best:
                self.best_miou = miou 
                self.best_epoch = epoch 
            self.writer.add_scalar("val/best_miou", self.best_miou, epoch)

            # SAVE METRICS.JSON
            self.metrics_log.append({
                "epoch": epoch, 
                "train_loss": float(train_loss),
                "train_ce_loss": float(ce_loss),
                "train_repa_cosine": float(repa_cosine) if self.args.use_repa else None,
                "repa_lambda":float(self._get_repa_lambda(epoch)) if self.args.use_repa else None,
                "train_repa_loss": float(repa_loss) if self.args.use_repa else None, # log only, when its a repa run 
                "val_loss": float(val_loss),
                "mean_accuracy":float(metrics["mean_accuracy"]), #Mean accuracy (averaged over all categories).
                "overall_accuracy": float(metrics["overall_accuracy"]), # Overall accuracy on all images.
                "mean_iou": float(metrics["mean_iou"]),
                "per_category_iou": [None if np.isnan(x) else float(x) for x in metrics["per_category_iou"]],
                "per_category_accuracy": [None if np.isnan(x) else float(x) for x in metrics["per_category_accuracy"]],
                "lr_groups": [float(group["lr"]) for group in self.optimizer.param_groups],
                "is_best": bool(is_best),
            })
            with open(self.run_dir / "metrics.json", "w") as file:
                json.dump({
                    "epochs": self.metrics_log,
                    "best_miou":self.best_miou,
                    "best_epoch": self.best_epoch 
                }, file, indent=2
                )
            self.writer.flush()

            # SAVE CKPT
            self._save_ckpt(epoch, metrics, is_best)

            # SAVE INTERMEDIATE CKPT (for later analysis)
            if self.args.save_ckpt_every is not None and (epoch+1) % self.args.save_ckpt_every == 0:
                self._save_intermediate_ckpt(epoch, metrics)
            
            print(f"Val_Loss:{val_loss: .4f}")
            print(f"Mean_iou: {metrics['mean_iou']: .4f}")
            print(f"Mean_accuracy: {metrics['mean_accuracy']:.4f}")

        self.writer.close()
        self._save_runtime()

    # PRIVATE METHODS
    def _train_one_epoch(self, epoch: int):
        print(f"Epoch: {epoch+1}/{self.args.epochs}")
        # TRAINMODE
        self.model.train() # set model to training mode; for dropout and batch normalization
        if self.repa_loss is not None: # if dropout or batchnorm is added
            self.repa_loss.train()

        total_run_loss = 0.0
        total_ce_loss = 0.0
        total_repa_loss = 0.0
        total_repa_cosine = 0.0

        # Compute current lambda
        repa_lambda = self._get_repa_lambda(epoch, decay_kind="cosine_decay")

        # ITERATE BATCH PER BATCH
        for idx, batch in enumerate(tqdm(self.train_loader)): # returns batch = {"image": tensor of x image-tensors, "mask": tensor of x masks}
            pixel_values = batch["image"].to(self.device) # (B,3,H,W) pixel_values is one batch tensor containing B images
            mask = batch["mask"].to(self.device)

            #  Channel dimension of masks (CE needs (B,H,W) --> albumentations sometimes returns (B,1,H,W))
            mask = mask[:, 0] if mask.ndim == 4 and mask.shape[1] == 1 else mask
            valid_pixel_mask = None

            # FORWARD PASS
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type): # (only forward pass computations run in 16 bit floats, the model weights stay in 32)
                if self.args.use_repa and repa_lambda > 0:
                    # load valid pixel mask
                    valid_pixel_mask = batch["valid_pixel_mask"].to(self.device)
                    if valid_pixel_mask.ndim == 4 and valid_pixel_mask.shape[1] == 1:
                        valid_pixel_mask = valid_pixel_mask[:, 0]

                    valid_pixel_mask = valid_pixel_mask.bool()

                    # ASSERTIONS - enable if wanted ====
                    # assert valid_pixel_mask.shape == mask.shape, (
                    #     f"Validity mask shape {valid_pixel_mask.shape} does not match segmentation mask shape {mask.shape}."
                    # )
                    # =========
                        
                    # Forward pass with REPA
                    logits, features = self.model(pixel_values, return_features = True) # logits are segmentation predictions, features are hidden states
                    ce_loss = self.criterion(logits, mask)

                    # Align student tokens of specific layer to final representation of teacher
                    last_layer_index = len(features['hidden_states']) - 1
                    # Allow normalized last layer representations
                    if (self.args.repa_student_layer == last_layer_index or self.args.repa_student_layer == -1 ) and self.args.normalized_last_student_layer:
                        student_patch_tokens = features["patch_tokens_last"]
                    else:
                        student_hidden_states = features['hidden_states'][self.args.repa_student_layer] # (B, N+1, D): get hidden states of specific layer # hidden_states is a tuple indexed from 0 to 12 for ViT base with 1-12 referring the patch embedding ouput after each trasnformer block 
                        student_patch_tokens = student_hidden_states[:, features["num_prefix_tokens"]:, :] # dropping CLS token: (B, N+1,D) --> (B, N, D) AND distill token for deit

                    repa = self.repa_loss(student_patch_tokens, pixel_values, valid_pixel_mask) # üixel_values for the teacher pass

                    repa_cosine = -repa.detach()

                    # Total loss = CE + λ · REPA
                    loss = ce_loss + repa_lambda * repa 
                    total_repa_loss += repa.item()
                    total_repa_cosine += repa_cosine.item()
                else:
                    # Forward Pass without REPA
                    logits = self.model(pixel_values) # (B, C, H, W) with C = 150
                    ce_loss = self.criterion(logits, mask)
                    loss = ce_loss
            self.scaler.scale(loss).backward() # compute gradients

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            clipping_params = list(self.model.parameters())
            if self.repa_loss is not None:
                clipping_params += list(self.repa_loss.proj.parameters()) # laernable projection MLP parameters
            torch.nn.utils.clip_grad_norm_(clipping_params, 1.0)

            # guard amp scaler against gradients overflow
            scale_pre = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_post = self.scaler.get_scale()
            if not scale_post < scale_pre:
                self.scheduler.step()

            # Log loss every 50 iterationes 
            if self.global_step % 50 == 0:
                self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
                self.writer.add_scalar("train/ce_loss_step", ce_loss.item(), self.global_step)
                if self.args.use_repa and repa_lambda > 0:
                    self.writer.add_scalar("train/repa_loss_step", repa.item(), self.global_step)
                    self.writer.add_scalar("train/repa_cosine_step", repa_cosine.item(), self.global_step)
                    self.writer.add_scalar("train/valid_pixel_fraction_step", valid_pixel_mask.float().mean().item(), self.global_step)
            self.global_step += 1
            total_run_loss += loss.item()
            total_ce_loss += ce_loss.item()

            if self.global_step == 1:
                print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

        # returning mean loss-values for whole epoch (Total-loss, total-ce-loss, total-repa-loss)
        return total_run_loss / len(self.train_loader), total_ce_loss / len(self.train_loader), total_repa_loss / len(self.train_loader), total_repa_cosine / len(self.train_loader)  # len(self.train_loader) is the number of batches in one epoch

    def _validation(self, epoch):
        with torch.no_grad(): # For efficiency: dont save gradients and computational graph
            # EVAL MODE
            self.model.eval()
            if self.repa_loss is not None: # added for consistency
                self.repa_loss.eval()

            metric = evaluate.load("mean_iou") #https://github.com/huggingface/evaluate/tree/main/metrics/mean_iou ; https://huggingface.co/docs/evaluate/v0.4.5/index 

            total_ce_sum = 0.0
            total_valid_pixels = 0

            # VALIDATION PER BATCH
            for batch in self.val_loader:
                pixel_values = batch["image"].to(self.device)
                mask = batch["mask"].to(self.device)

                # SLIDING WINDOW
                # see: https://github.com/rstrudel/segmenter/blob/master/segm/model/utils.py
                logits_windowed = sw.sliding_window_predict(
                    pixel_values, window_size=self.args.img_size,
                    window_stride=self.args.img_size - 32, window_batch_size=4, device=pixel_values.device.type,
                    num_classes=self.n_classes, model=self.model, patch_size=self.patch_size
                    )
                logits = F.interpolate( # resize to mask resolution
                    logits_windowed.unsqueeze(0), size=mask.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
                # Sum CE over all valid pixels int this image
                loss_sum = F.cross_entropy(
                    logits.float(), mask, ignore_index=255, reduction="sum",
                ) 

                valid_pixels = (mask != 255).sum()

                total_ce_sum += loss_sum.item()
                total_valid_pixels += valid_pixels.item()

                # Get predicted class per pixel   
                predicted = logits.argmax(dim=1)
                metric.add_batch(
                    predictions=predicted.cpu().numpy().astype(np.int32),
                    references=mask.cpu().numpy().astype(np.int32),
                )

        metrics = metric.compute(num_labels=self.n_classes, ignore_index=255, reduce_labels=False)
        return total_ce_sum / total_valid_pixels, metrics

    def _build_model(self) -> nn.Module:
        """
        Function to instantiate the student model. 
        Defines which student architecture to build based on --backbone-model.
        """
        return_features=self.args.use_repa
        if self.args.backbone_model.startswith("timm/vit"):
            model = TimmViT16_Segm(
                self.args.backbone_model, 
                self.args.num_classes, 
                return_features=return_features,
                head_type= self.args.head_type,
                pretrained = self.args.pretrained,
                drop_path_rate=self.args.drop_path,
                dropout_rate=self.args.dropout,
                attn_dropout_rate=self.args.att_dropout,
            ).to(self.device)
        elif self.args.backbone_model.startswith("scratch/vit"):
            model = ScratchViT14Segm(
                return_features=return_features,
                img_size=self.args.img_size,
                patch_size=self.args.patch_size,
                embedding_dim=self.args.embedding_dim,
                n_heads=self.args.n_heads,
                n_layers=self.args.n_layers,
                mlp_ratio=self.args.mlp_ratio,
                dropout=self.args.dropout,
                att_dropout=self.args.att_dropout,
                seg_head_type=self.args.head_type,
                drop_path=self.args.drop_path
            ).to(self.device)

        else:
            raise ValueError(f"Invalid --backbone-model argument entered: {self.args.backbone_model}")
        return model
  
    def _get_param_groups(self, module, lr):
        """
        _get_param_groups overwrites weight_decay with correct values
        """
        wd_params, no_wd_params = [], []
        for name, param in module.named_parameters():
            # 1D params are bias terms, ALyerNorm weight+bias , CLS token and posenc
            if param.ndim <= 1 or name.endswith(".bias") or "cls_token"in name or "position_embeddings" in name or "pos_embed" in name or "distillation_token" in name:
                no_wd_params.append(param)
            else :
                wd_params.append(param)
        return [
            {"params": wd_params, "lr": lr, "weight_decay": self.args.weight_decay},
            {"params": no_wd_params, "lr": lr, "weight_decay": 0.0},
        ]

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Builds AdamW optimizer with separate parameter groups (backbone, segmentation head, RepA projection),
        as the backbone, segmentation head and optional RepA projection are trainable by different LRs.
        Split into parameter groups by call of `_get_param_groups`:
            1) no_weight_decay: Bias terms, LayerNorm weights+bias, CLS token, Positional embeddings
                concretely: fc1.bias, fc2.bias, out.proj.bias, nn.LayerNorm, cls_token, position_embeddings
            2) weight_decay: all weight matrices
        """
        params = []
        if self.repa_loss is not None:
            params.extend(self._get_param_groups(self.model.backbone, self.args.lr_backbone))
            params.extend(self._get_param_groups(self.model.head, self.args.lr_head))
            params.extend(self._get_param_groups(self.repa_loss.proj, self.args.lr_repa))
        else:
            params.extend(self._get_param_groups(self.model.backbone, self.args.lr_backbone))
            params.extend(self._get_param_groups(self.model.head, self.args.lr_head))
        
        # _get_param_groups overwrites weight_decay with correct values, so no weight_decay param needs to be passed here 
        return torch.optim.AdamW(params, betas=(0.9, 0.999), eps=1e-8,)

    def _build_scheduler(self):
        """Unified building scheduler mathod with a lambda function."""
        total_steps = self.args.epochs * len(self.train_loader)
        warmup_steps = self.args.warmup_epochs * len(self.train_loader)
        decay_steps = total_steps - warmup_steps

        start_factor = 1e-3

        def lr_lambda(step: int) -> float:
            # Linear warmup
            if step < warmup_steps:
                progress = step / max(1, warmup_steps)
                return start_factor + progress * (1.0 - start_factor)
            # Decay
            decay_step = step - warmup_steps
            progress = min(decay_step / max(1, decay_steps), 1.0)

            # Poly Warmup (e.g. Strudel et al.)
            if self.args.scheduler == "poly_warmup":
                power = 0.9
                return (1.0 - progress) ** power
            elif self.args.scheduler == "constant_warmup":
                return 1.0

            raise ValueError(f"Scheduler {self.args.scheduler} is not supported.")

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _get_repa_lambda(self, epoch, decay_kind="cosine_decay"):
        """
        Returns Lambda value for Repa. 
        Applies Linear annealing if --repa_decay-start and --repa-decay-end are set.
        Inspired by Wang et al., "REPA Works Until It Doesn’t: 
        Early-Stopped, Holistic Alignment Supercharges Diffusion Training", NeurIPS, 2025 
        (https://neurips.cc/virtual/2025/loc/san-diego/poster/118901)
        
        Not tested yet!
        """
        if self.args.repa_decay_start is None and self.args.repa_decay_end is None:
            return self.args.repa_lambda
        if epoch < self.args.repa_decay_start:
            return self.args.repa_lambda
        if epoch >= self.args.repa_decay_end:
            return 0.0
        else:
            total_decay_epochs = self.args.repa_decay_end - self.args.repa_decay_start
            if decay_kind == "linear_decay":
                # Compute Linear Decay 
                decay_per_epoch = self.args.repa_lambda / total_decay_epochs
                current_lambda = self.args.repa_lambda - ((epoch - self.args.repa_decay_start) * decay_per_epoch)
            elif decay_kind == "cosine_decay":
                # Compute cosine decay
                current_relative_epoch = epoch - self.args.repa_decay_start
                current_lambda = (0.5) * self.args.repa_lambda * (1 + math.cos((current_relative_epoch / total_decay_epochs) * math.pi))
        return current_lambda

    def _infer_patch_size(self):
        """
        Infer patch size from model kind. Value error if inference is not possible.
            - Inference from AutoConfig: see https://huggingface.co/docs/transformers/model_doc/auto
        """
        if self.args.backbone_model.startswith("scratch/vit"):
            return self.args.patch_size
        elif self.args.backbone_model.startswith("timm/vit"):
             backbone_name = self.args.backbone_model.removeprefix("timm/")
             return timm.create_model(backbone_name, pretrained=False).patch_embed.patch_size[0]

        config = AutoConfig.from_pretrained(self.args.backbone_model)

        if hasattr(config, "patch_size"):
            return config.patch_size
        
        raise ValueError(f"Not able to infer patch size from config for backbone {self.args.backbone_model}")
    
    def _make_run_dirs(self, experiment_name:str):
        if self.args.resume is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            job_id = os.environ.get("SLURM_JOB_ID", "local")
            run_dir = self.args.runs_root / experiment_name / f"{job_id}_{timestamp}"
            ckpt_dir = run_dir / "checkpoints"
            log_dir = run_dir / "logs"
            run_dir.mkdir(parents=True, exist_ok=False)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
        else:
            resume_path = Path(self.args.resume)
            run_dir = resume_path.parent.parent
            ckpt_dir = resume_path.parent
            log_dir = run_dir / "logs"
        return run_dir, ckpt_dir, log_dir
    
    def _write_config(self):
        """
        Write config file into run folder. Saving Hyperparameters.
        """
        cfg = vars(self.args).copy() # creates dictionary, e.g. {"seed": 18, "lr_head": 0.0001, ...}.
        for key, val in cfg.items():
            if isinstance(val, Path):
                cfg[key] = str(val)

        config = {
            "config": cfg,
            "time": datetime.now().isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "model_info": {
                "patch_size": self.patch_size,
            },
            "transformations": {
                "train": {
                    
                    "reference_resize": {
                        "reference_scale": list(self.train_transforms.reference_scale),
                        "ratio_range": list(self.train_transforms.ratio_range),
                        "keep_ratio": self.train_transforms.keep_ratio,
                        "image_interpolation": "bilinear",
                        "mask_interpolation": "nearest",
                    },
                    "albumentation": self.train_transforms.transforms.to_dict(),
                    },
                "val": {
                    "inference": "single_scale_sliding_window",
                    "minimum_short_side": self.val_transforms.window_size,
                    "window_size": self.args.img_size,
                    "window_stride": self.args.img_size - 32,
                    "window_overlap": 32,
                    "window_batch_size": 4,
                    "normalization": self.val_transforms.normalize.to_dict(),
                },
            }
            # additional ones?
        }
        path = self.run_dir / "config.json"
        with open(path, "w") as file:
            json.dump(config, file, indent=2)
    
    def _save_runtime(self):
        """
        Append Runtime to config.
        """
        end_time = datetime.now()
        runtime = {
            "end_time": end_time.isoformat(),
            "runtime": (end_time-self.start_time).total_seconds(),
        }

        path = self.run_dir / "config.json"
        with open(path) as f:
            config = json.load(f)
        config.update(runtime)
        config.update({"best_miou": self.best_miou, "best_epoch": self.best_epoch})
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        
    def _set_seed(self):
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)

    def _load_ckpt(self, ckpt):
        self._resume_ckpt = ckpt 
        self.model.load_state_dict(ckpt["model"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.best_miou = ckpt["best_miou"]
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_epoch = ckpt["best_epoch"]
        self.global_step = ckpt["global_step"]
        if self.args.use_repa:
            self.repa_loss.proj.load_state_dict(ckpt["repa_loss"])
        self.start_epoch = ckpt["epoch"] + 1
        if "dataloader_generator_state" in ckpt:
            self.dataloader_generator.set_state(
                ckpt["dataloader_generator_state"].cpu()
            )

    def _save_ckpt(self, epoch, metrics, is_best=False):
        """
        Saving best and last model model weights to *.pth.
        """
        ckpt = {
            "epoch": epoch, 
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_miou": self.best_miou,
            "metrics": metrics, 
            "global_step": self.global_step,
            "best_epoch": self.best_epoch,
            "dataloader_generator_state": self.dataloader_generator.get_state(),
        }

        if self.repa_loss is not None:
            ckpt["repa_loss"] = self.repa_loss.proj.state_dict()
        
        ckpt["scheduler"] = self.scheduler.state_dict()

        torch.save (ckpt, self.ckpt_dir / "last.pth")
        if is_best:
            torch.save(ckpt, self.ckpt_dir / "best.pth")

    def _save_intermediate_ckpt(self, epoch, metrics):
        """Saves intermediate checkpoint for later analysis (e.G. representation probing or teacher/student similarity comparison)"""
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "metrics": metrics,
        }
        torch.save(ckpt, self.ckpt_dir / f"epoch_{epoch:03d}.pth")
