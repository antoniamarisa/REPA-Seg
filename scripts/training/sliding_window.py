"""
Single Scale Sliding Window Implementation used fo rinference in the validation Trainer 
and for qualitative analysis in scoring and rendering. 

This script follows the implementation of Studel et al., 2021:
Source code:
    https://github.com/rstrudel/segmenter/blob/master/segm/model/utils.py
Paper: 
    Strudel et al., "Segmenter: Transformer for Semantic Segmentation", CVF, 2021,
    https://openaccess.thecvf.com/content/ICCV2021/papers/Strudel_Segmenter_Transformer_for_Semantic_Segmentation_ICCV_2021_paper.pdf

"""

import torch
import numpy as np 
import torch.nn.functional as F

def sliding_window_predict(pixel_values, window_size, window_stride, window_batch_size,
                            device, num_classes, model, patch_size):
    """
    Single-image sliding-window inference.
    merged the methods of strudel "sliding_window" and merge_window" and "inference" into one.
    https://github.com/rstrudel/segmenter/blob/master/segm/model/utils.py
    - window batch size = 4
    """
    _, _, H, W = pixel_values.shape
    win_size = window_size
    h_anchors = torch.arange(0, H, window_stride)
    w_anchors = torch.arange(0, W, window_stride)
    h_anchors = [h.item() for h in h_anchors if h < H - win_size] + [H - win_size]
    w_anchors = [w.item() for w in w_anchors if w < W - win_size] + [W - win_size]

    # Sliding_window
    crops, anchors = [], []
    for ha in h_anchors:
        for wa in w_anchors:
            crops.append(pixel_values[:, :, ha:ha + win_size, wa:wa + win_size])
            anchors.append((ha, wa))
    # no flip, since here no flips are applied.
    crops = torch.cat(crops, dim=0)  # (n_windows, 3, win_size, win_size)

    # inference()
    num_windows = crops.shape[0]
    segm_maps= torch.zeros((num_windows, num_classes, win_size, win_size), device = pixel_values.device)
    with torch.autocast(device_type=device):
        for i in range(0, num_windows, window_batch_size):
            segm_maps[i : i + window_batch_size] = model(crops[i : i + window_batch_size]).float()

    # merge_windows
    logit = torch.zeros((num_classes, H, W), device=pixel_values.device)
    count = torch.zeros((1, H, W), device=pixel_values.device)
    for window, (ha, wa) in zip(segm_maps, anchors):
        logit[:, ha : ha + win_size, wa : wa + win_size] += window
        count[:, ha : ha + win_size, wa : wa + win_size] += 1
    return logit / count
    # interpolation is done in validation after this methos is called
    # no softmax calling since that is done in cross_entropy 