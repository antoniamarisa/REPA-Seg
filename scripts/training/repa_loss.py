"""
REPA loss computation for REPA-Seg

This implementation is based on the original REPA codebase: 

Original source implementation: 
    https://github.com/sihyun-yu/REPA 
    https://github.com/sihyun-yu/REPA/blob/main/loss.py

Paper: 
    S. Yu et al., "Representation Alignment for Generation: 
    Training Diffusion Transformers Is Easier Than You Think," ICLR, 2025. 
    (https://proceedings.iclr.cc/paper_files/paper/2025/file/d9e42b4d7163931f3689d6d6fbaa11d0-Paper-Conference.pdf)

Main adaptations are:
- adaptation to semantic segmentation 
- spatial alignment between student and teachers of different patch grid sizes 
- masking and weighting of padded image patches in computation of repa loss 

Supports Teacher Models: 
TESTED:
    - facebook/dinov2 
    - facebook/vit-mae
NOT TESTED:
    - facebook/dinov2-with-registers
    - facebook/dino-vit
    - google/vit
"""
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math 
from transformers import Dinov2Model, Dinov2WithRegistersModel, ViTModel, ViTMAEModel
from preprocessing.transforms import get_normalization

class REPALoss(nn.Module): # https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html (every nn.Module is a Callable)
    def __init__(
            self, student_dim,
            teacher = "facebook/dinov2-base",
            teacher_layer = -1, normalized_last_teacher_layer = True,
            repa_proj = "3l-mlp-silu", student_backbone=None
            ):
        super().__init__()

        # LOAD PRETRAINED TEACHER
        if teacher.startswith("facebook/dinov2-with-registers"):
            self.teacher = Dinov2WithRegistersModel.from_pretrained(teacher)
        elif teacher.startswith("facebook/dinov2"):
            self.teacher = Dinov2Model.from_pretrained(teacher)
        elif teacher.startswith("facebook/dino-vit"):
            self.teacher = ViTModel.from_pretrained(teacher)
        elif teacher.startswith("google/vit"): # Supervised!
             self.teacher = ViTModel.from_pretrained(teacher)
        elif teacher.startswith("facebook/vit-mae"):
            self.teacher = ViTMAEModel.from_pretrained(teacher)
            self.teacher.config.mask_ratio = 0.0 # disable random patch masking
        else: 
            raise ValueError(f"unsuppoted type of the repa-teacher model: {teacher}")
        self.teacher.eval()

        # PREFIX TOKENS: 
        # normally 1 for cls, but additional registers for dinov2 /w registers
        if isinstance(self.teacher, Dinov2WithRegistersModel):
            self.teacher_num_prefix_tokens = 1 + self.teacher.config.num_register_tokens
        else:
            self.teacher_num_prefix_tokens = 1

        # FREEZE TEACHER
        for parameter in self.teacher.parameters():
            parameter.requires_grad= False 

        # NORMALIZATION MISMATCH HANDLING
        # necessray for combining AugReg student with dinov2 teacher
        student_mean, student_std = get_normalization(student_backbone)
        teacher_mean, teacher_std = get_normalization(teacher)
        # assure non learnable state of these values
        self.register_buffer("student_mean", torch.tensor(student_mean).view(1, 3, 1, 1))
        self.register_buffer("student_std", torch.tensor(student_std).view(1, 3, 1, 1))
        self.register_buffer("teacher_mean", torch.tensor(teacher_mean).view(1, 3, 1, 1))
        self.register_buffer("teacher_std", torch.tensor(teacher_std).view(1, 3, 1, 1))
        
        teacher_dim = self.teacher.config.hidden_size 
        self.normalized_last_teacher_layer = normalized_last_teacher_layer

        # LEARNABLE PROJECTION
        # maps student tokens into teacher-feature-space
        # 1) Linear projection
        if repa_proj == "linear":
            self.proj = nn.Linear(student_dim, teacher_dim)
        # 2) 3-layer MLP with SiLU (Yu et al., 2025)
        elif repa_proj == "3l-mlp-silu":
            self.proj = nn.Sequential(
                nn.Linear(student_dim, teacher_dim),
                nn.SiLU(), 
                nn.Linear(teacher_dim, teacher_dim),
                nn.SiLU(),
                nn.Linear(teacher_dim, teacher_dim)
            )
        else:
            raise ValueError(f"No valid repa projection selected: {repa_proj}")

        # INITIALIZE REPA PROJECTION WEIGHTS 
        # matches Yu et al. (https://github.com/sihyun-yu/REPA/blob/main/models/sit.py)
        self.proj.apply(self._init_proj_weights)

        self.teacher_layer = teacher_layer

    def _init_proj_weights(self, module):
        """ this follow the initialization principle in yu et al., that repa is samely initialized as the underlying student model. 
        https://github.com/sihyun-yu/REPA/blob/main/models/sit.py
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    # Overwriting of train so the child modules including the teacher dont get set back to train mode
    def train(self, mode = True):
        """
        This method overwrites train() as the teacher should stay in eval mode, even if projection gets trained.
        """
        super().train(mode)
        self.teacher.eval() # teacher stays in eval mode
        return self
    
    @torch.no_grad()
    def teacher_features(self, images):
        """
        Computes teacher representations for one batch of images.
            - images (normalized for the student) are re-normalized to the teacher's own pretraining 
                statistics, then passed thorugh the teacher model (e. g. DINOv2) -> necessary for e.g. AugReg and DINOv2 combination
            - hidden states from the selected teacher layer are returned and CLS token is dropped
        No gradients get stored, as the teacher is not trained.
        """
        # RE-NORMALIZE
        # per pixel renormalization since teacher features in the argument here are already student values normalized (and maybe augreg normalized)
        images = images * self.student_std + self.student_mean # undo student normaliazion 
        images = (images - self.teacher_mean) / self.teacher_std # reapply normalization with teacher values 
        
        # FORWARD PASS
        if isinstance(self.teacher, ViTMAEModel): # NOT TESTED YET 
            # prevent that the batches get randomly shuffled, otherwise student ptch i would be aligned to a random teacher patch 
            B, C, H, W = images.shape
            patch_size = self.teacher.config.patch_size
            n_patches  = (H//patch_size) * (W//patch_size)
            # determnistic identity-order noise
            noise = torch.arange(n_patches, device=images.device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
            out = self.teacher(pixel_values=images, noise=noise, output_hidden_states=True, interpolate_pos_encoding=True)
        else:
            # for DINOv2-large (24 tarsnformer blocks) out is a tuple of 25 tensors with 0 as the patch embedding output, and 1-24 fter each trasnformer block
            out = self.teacher(pixel_values = images, output_hidden_states = True,interpolate_pos_encoding=True,) 
            # out is BaseModelOutputWithPooling https://huggingface.co/transformers/v4.1.1/main_classes/output.html#basemodeloutputwithpooling

        prefix = self.teacher_num_prefix_tokens
        if self.normalized_last_teacher_layer == False:
            return out.hidden_states[self.teacher_layer][:,prefix:,:] # droppping CLS token (and potentially register tokens)
        elif self.normalized_last_teacher_layer == True and (self.teacher_layer == -1):
            return out.last_hidden_state[:,prefix:,:]
        else:
            raise ValueError(f"Unsupported combination of chosen teacher layer: {self.teacher_layer} and last layer output normalized = {self.normalized_last_teacher_layer}")

    def forward(self, student_patch_tokens, images, valid_pixel_mask=None):
        """
        Computation of REPA loss for one batch. Gets called with repa_loss() once per training batch.
            1) teacher patch features get extracted from the frozen teacher model
                - if necessary, they get interpolated to the student's grid or the student patches get upsample to the teachers grid
            2) student patch tokens are projected into the teacher feature dimension
            3) L2 normalization
            4) Loss is compoted as the negative mean cosine similarity between corresponding teacher and projected student patch teachers

        Implementation is based on the code base of https://github.com/sihyun-yu/REPA/blob/main/loss.py
        """

        # COMPUTE TEACHER FEATURES
        z = self.teacher_features(images) # z.shape = (B, N_t, D_t) <=> (batch_size, sequence_length, hidden_size)
        # DELETE after running
        assert z.ndim == 3
        assert student_patch_tokens.ndim == 3
        # ========

        # GET SHAPES
        B, N_s, _ = student_patch_tokens.shape # student shape, N_s = number of student-patch-tokens
        N_t = z.shape[1] # N_t = number of teacher-patch-tokens

        # PROJECT student representations into teacher feature dimension
        z_tilde = self.proj(student_patch_tokens) # (B, N_s, D_s) -> (B, N_s, D_t)

        h_s = w_s = int(math.sqrt(N_s))
        h_t = w_t = int(math.sqrt(N_t))

        if N_t == N_s:
            h_out, w_out, N_out = h_s, w_s, N_s
        
        # INTERPOLATION
        elif N_t > N_s:
            # AM-RADIO (Ranzinger et al.): upsample student to match the finer teacher grid
            z_tilde = z_tilde.reshape(B, h_s, w_s, -1).permute(0, 3, 1, 2)
            z_tilde = F.interpolate(z_tilde, size=(h_t, w_t), mode="bilinear", align_corners=False)
            z_tilde = z_tilde.permute(0, 2, 3, 1).reshape(B, N_t, -1)
            h_out, w_out, N_out = h_t, w_t, N_t

        else:
            # upsample teacher to student grid
            z = z.reshape(B, h_t, w_t, -1).permute(0,3,1,2) # (B, N_t, D_t) -> (B, h_t, w_t, D_t) -> (B, D_t, h_t, w_t)
            # Interpolate teacher-feature-map to student-grid (e. g. (B, D_t, 32, 32) -> (B, D_t, 28, 28))
            z = F.interpolate(z, size=(h_s,w_s), mode = "bilinear", align_corners=False) 
            # Back to token shape — teacher token number now equals student token number
            z = z.permute(0,2,3,1).reshape(B,N_s, -1) # (B, D_t, h_s, w_s) -> (B, h_s, w_s, D_t) -> (B, N_s, D_t)
            h_out, w_out, N_out = h_s, w_s, N_s

        # NORMALIZE
        z = F.normalize(z, dim = -1) # l2-normalize each patch vector to length 1 for cosine similarity
        z_tilde = F.normalize(z_tilde, dim = -1)

        # DELETE
        assert z.shape == z_tilde.shape

        similarity = (z*z_tilde).sum(dim=-1)

        if valid_pixel_mask is None:
            return -similarity.mean() # (12, 784, 768) * (12, 784, 768) -> (12, 784, 768) -> (12,184) -> mean over all patches in all images per batch 
        else:
            # ADDITION of weighting of each student patch by its fraction of non-padding image pixels
            # Addition to yu et al, since they havent done preprocessing that produced padding
            valid_fraction = F.adaptive_avg_pool2d(
                valid_pixel_mask.float().unsqueeze(1),
                output_size=(h_out, w_out)
            )
            weights = valid_fraction.reshape(B, N_out)

            return -(similarity * weights).sum() / weights.sum().clamp_min(1e-6)