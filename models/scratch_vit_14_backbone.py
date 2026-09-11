"""
Vision Transformer segmentation model: Custom scratch ViT-14/S.

The backbone follows the ViT architecture introduced by:
    Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    ICLR 2021. https://arxiv.org/abs/2010.11929

The implementation follows standard ViT formulations, based on the following reference implementations:
    Google Research ViT:
    https://github.com/google-research/vision_transformer
    Hugging Face ViT:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py

Stochastic depth is implemented using timm's DropPath:
    https://github.com/huggingface/pytorch-image-models

Segmentation head follow the linear decoder in Segmenter:
    Strudel et al., "Segmenter: Transformer for Semantic Segmentation", ICCV 2021.
    Paper:
    https://arxiv.org/abs/2105.05633
    Code:
    https://github.com/rstrudel/segmenter/blob/master/segm/model/decoder.py

The default backbone corresponds to a ViT-S/14-style architecture:
    depth:          12
    embedding dim:  384
    MLP dim:        1536
    attention heads: 6
    patch size:     14
Model is trained from random initialization.
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F

from torchinfo import summary

from timm.layers import DropPath

def init_weights(module, std=0.02):
    """
    Initialize weights for all vit blocks.
    Applies to Linear Layers, Conv2d patch projection, MHSA, LayerNorm.
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias) # set zeros for bias term
    elif isinstance(module, nn.Conv2d):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias) # set zeros for bias term
    elif isinstance(module, nn.MultiheadAttention):
        # Q,K,V are stored together
        if module.in_proj_weight is not None:
            nn.init.trunc_normal_(module.in_proj_weight, std=std)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias) # set zeros for bias term

class PatchEmbedding(nn.Module):
    """
    Split an image into non-overlapping patches and project each patch to the ViT embedding dimension.
    """
    def __init__(self, img_size=448, patch_size=14, n_channels=3, embedding_dim=384): # embedding_dim =  384 for small 
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisable by patch size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_patches = (img_size // patch_size) ** 2 # 32*32 = 1024 patch tokens

        # Conv2D projection (like HF, as this is mathematically equivalent and more efficient)
        self.projection = nn.Conv2d(n_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values): # x.shape = (batch_size, n_channels, img_size, img_size)
        B, C, H, W = pixel_values.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Input size must dividable by patch size.")

        if C != self.n_channels:
            raise ValueError(
                f"Channel dimension of the pixel values do not match with the one set in the configuration. Expected {self.n_channels} but got {C}."
            )
        proj = self.projection(pixel_values).flatten(2)
        return proj.transpose(1,2)

class MLP(nn.Module):
    """
    MLP.
    mlp_ratio: MLP expands the model's main embedding_dim by a factor of mlp_ratio (mlp_hidden_dim = mlp_ratio * embedding_dim)
    """
    def __init__(self, embedding_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        mlp_hidden_dim = int(mlp_ratio*embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, mlp_hidden_dim)
        self.gelu = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(mlp_hidden_dim,embedding_dim)
        self.drop2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block.
    Follows Dosovitskiy (LayerNorm, MHSA, residual connection, LayerNorm, MLP, residual connection).
    """
    def __init__(self, embedding_dim, n_heads=6, mlp_ratio = 4.0, att_dropout = 0.0, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.mhsa = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads, dropout=att_dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_ratio, dropout)
        self.drop2 = nn.Dropout(dropout)

        self.drop_path1= (DropPath(drop_path) if drop_path > 0.0 else nn.Identity())
        self.drop_path2= (DropPath(drop_path) if drop_path > 0.0 else nn.Identity())
    
    def forward(self, x):
        y = self.layer_norm1(x) # normalize across last (emebdding) dimension of input vector
        att_output, _ = self.mhsa(y, y, y, need_weights=False) # need_weights=False for efficiency, interesting for inspecting attention maps after training 

        x = x + self.drop_path1(self.drop1(att_output))

        y = self.layer_norm2(x)
        y = self.mlp(y) 

        x = x + self.drop_path2(y)
        return x 

class ViT14(nn.Module): 
    """
    Vision Transformer (ViT) model.
    """
    def __init__ (
            self, return_features=False,
            img_size=518, # or also 448 to compare?
            patch_size=14, n_channels=3, embedding_dim=384, n_layers=12,
            n_heads=6,mlp_ratio=4.0, att_dropout=0.0, # 0.1?
            dropout=0.1, drop_path=0.0,
        ):
        super().__init__()

        self.return_features = return_features

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.embedding_dim = embedding_dim

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.att_dropout = att_dropout
        self.dropout = dropout
        self.drop_path = drop_path

        self.patch_embedding = PatchEmbedding(self.img_size, self.patch_size, self.n_channels, self.embedding_dim)

        # CLS + POSITIONAL ENCODING INITIALIZATION
        self.cls_token = nn.Parameter(torch.empty(1,1,self.embedding_dim)) # inital CLS token (1,1,384)
        self.position_embeddings = nn.Parameter(torch.empty(1,1+self.patch_embedding.n_patches, self.embedding_dim)) # inital pos embedding tesor (1, 1+1369, 384)

        self.pos_dropout = nn.Dropout(self.dropout)        

        # Linearly increse drop path rates to rather randomly skip later layers 
        drop_path_rates = torch.linspace(0, self.drop_path, self.n_layers).tolist()
                
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dim=self.embedding_dim,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                att_dropout=self.att_dropout,
                dropout=self.dropout,
                drop_path = drop_path_rates[i]
            )
            for i in range(self.n_layers)
        ])
        
        self.norm = nn.LayerNorm(embedding_dim)

        # INITIALIZE WEIGHTS
        # If ViT14 is used alone
        # self.apply(lambda module: init_weights(module))

    def _interpolate_pos_encoding(self, grid_h, grid_w):
        """
        Interpolation of positional encoding, cause now for validation images, the image sizes can be non-square or larger due
        to changed to preserving aspect ration (Segmenter style)

        The patch positional encoding are reshaped to their original 2D
        training grid, interpolated bicubically to the current patch grid, and flattened back into a token sequence. 
        The CLS positional encoding remains unchanged.

        This follows the standard 2D positional-embedding interpolation approach used for Vision Transformers (Dosovitskiy et al., 2021)
        and in DINO/DINOv2 implementations.
        """
        trained_h = trained_w = self.img_size // self.patch_size  # number of patches along one dimension of the training grid
        if grid_h == trained_h and grid_w == trained_w:
            return self.position_embeddings # No interpolation needed 

        cls_pos   = self.position_embeddings[:, :1, :]   # (1, 1, D)
        patch_pos = self.position_embeddings[:, 1:, :]   # (1, trained_h*trained_w, D)
        embedding_dim = patch_pos.shape[-1]

        # reshape to spatial grid, bicubic-interpolate, flatten back
        patch_pos = patch_pos.reshape(1, trained_h, trained_w, embedding_dim).permute(0, 3, 1, 2)  # (1, trained_h * trained_w, D) -> (1, D, trained_h, trained_w)
        patch_pos = F.interpolate(
            patch_pos.float(), size=(grid_h, grid_w), mode="bicubic", align_corners=False
        ).to(self.position_embeddings.dtype)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, embedding_dim)   # (1,D,grid_h,grid_w) -> (1, grid_h*grid_w, D)

        return torch.cat([cls_pos, patch_pos], dim=1)   # (1, 1+grid_h*grid_w, D)

    def forward(self, pixel_values, return_features = False):
        """
        Return patch_tokens to the SegmentationHead.
        - pixel_valeus: (B, C, H, W) — H and W must be divisible by patch_size but no longer need to equal img_size.
        if return features is enabled: 
        - Returns all hidden_states with the last layer before layernorm is applied (similar to HF implementation)
        - Returns last hidden_state with Layer Norm (similar to HF implementation)
        otherwise only Layer normalized last_hidden_state is returned
        """
        B = pixel_values.shape[0]
        H, W = pixel_values.shape[-2:]
        grid_h, grid_w = H // self.patch_size, W // self.patch_size

        x = self.patch_embedding(pixel_values) # (B, N, D)

        # Make the learnable CLS token available for every image in the batch
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, D) (cls_token.shape=(1, 1, 384) -1 keeps original size in the dimensionion 
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self._interpolate_pos_encoding(grid_h, grid_w)
        x = self.pos_dropout(x)

        # pass thorugh all blocks
        hidden_states = [x]
        for block in self.transformer_blocks:
            x = block(x)
            hidden_states.append(x)
        
        last_hidden_state = self.norm(x)
        if return_features == True:
            return last_hidden_state, hidden_states
        else:
            return last_hidden_state

class ScratchViT14Segm(nn.Module):
    def __init__(self,  
            return_features=False, img_size=448,
            patch_size=14, n_channels=3, n_classes=150,  embedding_dim=384, 
            n_layers=12, n_heads=6, mlp_ratio=4.0, att_dropout=0.0, 
            dropout=0.0, seg_head_type="linear", drop_path= 0.05,
        ):
        super().__init__()
        self.backbone = ViT14(return_features=return_features, img_size=img_size, patch_size=patch_size, 
                              n_channels=n_channels, embedding_dim=embedding_dim, n_layers= n_layers, 
                              n_heads= n_heads, mlp_ratio= mlp_ratio,att_dropout= att_dropout,dropout= dropout, drop_path=drop_path)
        self.return_features = return_features
        self.patch_size = patch_size
    
        # HEAD TYPE
        if seg_head_type == "mlp":
            # Simple 2-layer MLP with GELU
            self.head = nn.Sequential(
                nn.Linear(embedding_dim,embedding_dim),
                nn.GELU(),
                # nn.Dropout(0.1)
                nn.Linear(embedding_dim, n_classes)
            )
        elif seg_head_type == "conv": # (SETR, Zheng et al., 2021; DPT, Ranftl et al., 2021)
            raise NotImplementedError(f"Conv head type not implemented yet")
            # TODO
        else: 
            # Segmenter (Strudel et al., 2021)
            self.head = nn.Linear(embedding_dim, n_classes) # Decoder: Segmentation head: gets token vector of size dim and outputs a logit vector of size n_classes

        # INITIALIZE WEIGHTS
        # 1) INIT MODULES: Walks the entire Module tree and calls init_weights(module) on every nn.Module
        self.apply(init_weights)
        # 2) INIT CLS + POS_ENC:
        #  initialize weights in HF-style: CLS normal, position normal; Google original-style: CLS zero, position normal
        nn.init.trunc_normal_(self.backbone.cls_token, std=0.02)
        nn.init.trunc_normal_(self.backbone.position_embeddings, std=0.02)

    def forward(self, pixel_values, return_features = False): 
        """
        Forward pass.
        Returns dict of:
            - patch_tokens_last: unnormalized patch tokens
            - hidden_states: unnormalized
            - last_hidden_state: normalized
        """
        # FORWARD PASS
        if return_features:
            last_hidden_state, hidden_states = self.backbone(pixel_values = pixel_values, return_features=True) # (B, 1+N, D)
        else:
            last_hidden_state = self.backbone(pixel_values=pixel_values)
        patch_tokens = last_hidden_state[:,1:, :] # (B, N, D) --> CLS token dropped, only keep patch tokens

        B, N, D = patch_tokens.shape # (B = batch_size, N = number of patches (for 448x448 and patch size 14 it is 32*32 = 1024), D = hidden dimension=384 for vit-s)

        # Determine grid-size N = (H/P) * (W/P)
        H,W = pixel_values.shape[-2], pixel_values.shape[-1] # e.g. pixel_values.shape = (12, 3, 448, 448) return in H = 448 and W = 448
        
        # Compute grid dimensions 
        h = H // self.patch_size # e.g. 448 // 14 =  --> 32x32 
        w = W // self.patch_size 
        assert h*w == N, f"Token mismatch"

        # SEGMENTATION HEAD TO PREDICT MASKS
        # apply head on each patch token
        logits_patch= self.head(patch_tokens) # (B, N, C) 
        logits_patch = logits_patch.transpose(1,2).reshape(B, -1, h, w) # (B, N, C) -> (B, C, N) -> (B, C , h , w)

        # Upsample to full resolution from flat sequence intoo rthe 2d spatial grid: 32x32 -> 448x448
        logits = F.interpolate(logits_patch, size = (H,W), mode = "bilinear", align_corners=False) # change mode? -> no
        
        if return_features:
            return logits, { 
                "patch_tokens_last": patch_tokens, # without cls token
                "hidden_states": hidden_states, # all hidden states, unnormalized
                "last_hidden_state": last_hidden_state, # after LayerNorm
                "num_prefix_tokens": 1, # cls token 
            }
        return logits
    
if __name__ == '__main__':
    model = ScratchViT14Segm(img_size=448, patch_size=14, return_features=False)
    summary(
        model,
        input_size=(1,3,448,448),
        col_names=('input_size', 'output_size', 'num_params'),
    )

    """"
========================================================================================================================
Layer (type:depth-idx)                        Input Shape               Output Shape              Param #
========================================================================================================================
ScratchViT14Segm                              [1, 3, 448, 448]          [1, 150, 448, 448]        --
├─ViT14: 1-1                                  --                        [1, 1025, 384]            393,984
│    └─PatchEmbedding: 2-1                    [1, 3, 448, 448]          [1, 1024, 384]            --
│    │    └─Conv2d: 3-1                       [1, 3, 448, 448]          [1, 384, 32, 32]          226,176
│    └─Dropout: 2-2                           [1, 1025, 384]            [1, 1025, 384]            --
│    └─ModuleList: 2-3                        --                        --                        --
│    │    └─TransformerEncoderBlock: 3-2      [1, 1025, 384]            [1, 1025, 384]            1,774,464
│    │    └─TransformerEncoderBlock: 3-3      [1, 1025, 384]            [1, 1025, 384]            1,774,464
│    │    └─TransformerEncoderBlock: 3-4      [1, 1025, 384]            [1, 1025, 384]            1,774,464
│    │    └─TransformerEncoderBlock: 3-5      [1, 1025, 384]            [1, 1025, 384]            1,774,464
│    │    └─TransformerEncoderBlock: 3-6      [1, 1025, 384]            [1, 1025, 384]            1,774,464
│    │    └─TransformerEncoderBlock: 3-7      [1, 1025, 384]            [1, 1025, 384]            1,774,464
│    │    └─TransformerEncoderBlock: 3-8      [1, 1025, 384]            [1, 1025, 384]            1,774,464
│    │    └─TransformerEncoderBlock: 3-9      [1, 1025, 384]            [1, 1025, 384]            1,774,464
│    │    └─TransformerEncoderBlock: 3-10     [1, 1025, 384]            [1, 1025, 384]            1,774,464
│    │    └─TransformerEncoderBlock: 3-11     [1, 1025, 384]            [1, 1025, 384]            1,774,464
│    │    └─TransformerEncoderBlock: 3-12     [1, 1025, 384]            [1, 1025, 384]            1,774,464
│    │    └─TransformerEncoderBlock: 3-13     [1, 1025, 384]            [1, 1025, 384]            1,774,464
│    └─LayerNorm: 2-4                         [1, 1025, 384]            [1, 1025, 384]            768
├─Linear: 1-2                                 [1, 1024, 384]            [1, 1024, 150]            57,750
========================================================================================================================
Total params: 21,972,246
Trainable params: 21,972,246
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 245.86
========================================================================================================================
Input size (MB): 2.41
Forward/backward pass size (MB): 272.02
Params size (MB): 57.93
Estimated Total Size (MB): 332.36
========================================================================================================================
    """