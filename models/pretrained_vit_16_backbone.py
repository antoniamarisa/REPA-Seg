"""
Vision Transformer segmentation model: AugReg-pretrained Vit-S/16.

The backbone uses the timm checkpoint `vit_small_patch16_224.augreg_in21k_ft_in1k`, 
pretrained on ImageNet-21k and fine-tuned on ImageNet-1k:
https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k

timm documentation and implementation:
https://huggingface.co/docs/timm/
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py


Segmentation head follow the linear decoder in Segmenter:
    Strudel et al., "Segmenter: Transformer for Semantic Segmentation", ICCV 2021.
    Paper:
    https://arxiv.org/abs/2105.05633
    Code:
    https://github.com/rstrudel/segmenter/blob/master/segm/model/decoder.py
"""

import torch.nn as nn 
import torch.nn.functional as F
import timm

from torchinfo import summary

class TimmViT16_Segm(nn.Module): 
    """
    Pretrained AugReg Vit-S/16 model class downloaded from  HF with timm. 
    """
    def __init__ (self, backbone:str, n_classes: int, return_features:bool=False, 
                  head_type="linear", pretrained=True, drop_path_rate=0.0, dropout_rate=0.0, attn_dropout_rate=0.0):
        super().__init__()

        # PRETRAINED STUDENT BACKBONE MODEL
        if pretrained:
             backbone_name = backbone.removeprefix("timm/")
             self.backbone = timm.create_model(backbone_name, pretrained=True, 
                                               num_classes=0, dynamic_img_size=True, drop_path_rate=drop_path_rate, 
                                               proj_drop_rate=dropout_rate, pos_drop_rate=dropout_rate, attn_drop_rate=attn_dropout_rate,)
        else:
            raise NotImplementedError(f"Model instanciation for pretrained=False for {backbone} not implemented")
        
        self.return_features = return_features
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.num_prefix_tokens=self.backbone.num_prefix_tokens

        dim = self.backbone.embed_dim

        # HEAD TYPE
        if head_type == "mlp":
            # Simple 2-layer MLP with GELU
            self.head = nn.Sequential(
                nn.Linear(dim,dim),
                nn.GELU(),
                # nn.Dropout(0.1)
                nn.Linear(dim, n_classes)
            )
        elif head_type == "conv": # (SETR, Zheng et al., 2021; DPT, Ranftl et al., 2021)
            raise NotImplementedError(f"Conv head type not supported yet")
            # TODO
        else: 
            # Segmenter (Strudel et al., 2021)
            self.head = nn.Linear(dim, n_classes) # Decoder: Segmentation head: gets token vector of size dim and outputs a logit vector of size n_classes


    def _hidden_states(self, pixel_values):
        """
        returns hidden states (B, num_prefix_tokens+N, D) tensors, onw per layer with 0 index being the pre-block embedding 
        and index i the output after transformer block i. It collects the hidden states by hand.
        1. embedding output is collected.
        2. get_intermediate_layers runs the model once and returns depth entries pre final layer norm 
        """
        x = self.backbone.patch_embed(pixel_values)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)

        hidden_states = [x]

        for block in self.backbone.blocks:
            x = block(x)
            hidden_states.append(x)

        return tuple(hidden_states)

    def forward(self, pixel_values, return_features = False, out_size = None): 
        """
        Run the timm ViT-based segmentation model on one batch of images.
            1) Images get passed thorugh a pretrained ViT backbone to obtain patch-level token representations
            2) CLS token is dropped 
            3) segmentation head maps each patch token to class logits.
            4) Patch logits get reshoaed into a low-resolution spatial grid
            5) and then bilinearly upsampled to the target item size
        If return_features is True, the function also returns the hidden states for RepA.
        Args:
            - pixel_vlaues is a float tensor (B,3,H,W) as required by the Huggingface ViT backbone
            - out_size (H,W): optionale targetsize of output logits for reshape

        """
        # FORWARD PASS
        if return_features:
            hidden_states = self._hidden_states(pixel_values)
            toks = self.backbone.norm(hidden_states[-1]) # final after layer normalization
        else:
            toks = self.backbone.forward_features(pixel_values)

        # Determine grid-size N = (H/P) * (W/P)
        if out_size is None:
            H,W = pixel_values.shape[-2], pixel_values.shape[-1] # e.g. pixel_values.shape = (12, 3, 448, 448) return in H = 448 and W = 448
        else:
            H, W = out_size
        
        # Compute grid dimensions 
        h = H // self.patch_size # e.g. 448 // 16 = 28 --> 28x28 grid of 784 tokens
        w = W // self.patch_size 

        # Drop the prefix cls token
        patch_tokens = toks[:, self.num_prefix_tokens:, :] # (B, N, D)
        B, N, D = patch_tokens.shape
        assert h*w == N, f"Token mismatch"

        # SEGMENTATION HEAD TO PREDICT MASKS
        # apply head on each patch token
        logits_patch= self.head(patch_tokens) # (B, N, C) 
        logits_patch = logits_patch.transpose(1,2).reshape(B, -1, h, w) # (B, N, C) -> (B, C, N) -> (B, C , h , w)

        # Upsample to full resolution from flat sequence into the 2d spatial grid: 28x28 -> 448x448
        logits = F.interpolate(logits_patch, size = (H,W), mode = "bilinear", align_corners=False) # change mode -> no, should be alright
        
        if return_features:
            return logits, { 
                "patch_tokens_last": patch_tokens,
                "hidden_states": hidden_states,
                "num_prefix_tokens": self.num_prefix_tokens,
            }
        return logits

if __name__ == '__main__':
    model = TimmViT16_Segm(backbone="vit_small_patch16_224.augreg_in21k_ft_in1k", n_classes=150, return_features=False)
    summary(
        model,
        input_size=(1,3,448,448),
        col_names=('input_size', 'output_size', 'num_params'),
    )

"""
========================================================================================================================
Layer (type:depth-idx)                        Input Shape               Output Shape              Param #
========================================================================================================================
TimmViT16_Segm                                [1, 3, 448, 448]          [1, 150, 448, 448]        --
├─VisionTransformer: 1-1                      --                        --                        76,032
│    └─PatchEmbed: 2-1                        [1, 3, 448, 448]          [1, 28, 28, 384]          --
│    │    └─Conv2d: 3-1                       [1, 3, 448, 448]          [1, 384, 28, 28]          295,296
│    │    └─Identity: 3-2                     [1, 28, 28, 384]          [1, 28, 28, 384]          --
│    └─Dropout: 2-2                           [1, 785, 384]             [1, 785, 384]             --
│    └─Identity: 2-3                          [1, 785, 384]             [1, 785, 384]             --
│    └─Identity: 2-4                          [1, 785, 384]             [1, 785, 384]             --
│    └─Sequential: 2-5                        [1, 785, 384]             [1, 785, 384]             --
│    │    └─Block: 3-3                        [1, 785, 384]             [1, 785, 384]             1,774,464
│    │    └─Block: 3-4                        [1, 785, 384]             [1, 785, 384]             1,774,464
│    │    └─Block: 3-5                        [1, 785, 384]             [1, 785, 384]             1,774,464
│    │    └─Block: 3-6                        [1, 785, 384]             [1, 785, 384]             1,774,464
│    │    └─Block: 3-7                        [1, 785, 384]             [1, 785, 384]             1,774,464
│    │    └─Block: 3-8                        [1, 785, 384]             [1, 785, 384]             1,774,464
│    │    └─Block: 3-9                        [1, 785, 384]             [1, 785, 384]             1,774,464
│    │    └─Block: 3-10                       [1, 785, 384]             [1, 785, 384]             1,774,464
│    │    └─Block: 3-11                       [1, 785, 384]             [1, 785, 384]             1,774,464
│    │    └─Block: 3-12                       [1, 785, 384]             [1, 785, 384]             1,774,464
│    │    └─Block: 3-13                       [1, 785, 384]             [1, 785, 384]             1,774,464
│    │    └─Block: 3-14                       [1, 785, 384]             [1, 785, 384]             1,774,464
│    └─LayerNorm: 2-6                         [1, 785, 384]             [1, 785, 384]             768
├─Linear: 1-2                                 [1, 784, 384]             [1, 784, 150]             57,750
========================================================================================================================
Total params: 21,723,414
Trainable params: 21,723,414
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 252.86
========================================================================================================================
Input size (MB): 2.41
Forward/backward pass size (MB): 324.08
Params size (MB): 86.59
Estimated Total Size (MB): 413.08
========================================================================================================================
"""