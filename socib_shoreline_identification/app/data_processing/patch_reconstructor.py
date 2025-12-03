import numpy as np
import torch
from typing import List, Dict

class PatchReconstructor():

    @staticmethod
    def combine_patches(output: torch.Tensor, n_classes: int, padding: List, patches: dict, patch_size: tuple = (256, 256), stride: tuple = (128, 128), method: str = 'avg') -> torch.Tensor:

        n_rows = max(patch["row"] for patch in patches) + 1
        n_cols = max(patch["col"] for patch in patches) + 1

        if patch_size[0] == stride[0]:
            orig_h = n_rows * patch_size[0]
        else:
            orig_h = n_rows * (patch_size[0] - stride[0]) + stride[0]

        if patch_size[1] == stride[1]:
            orig_w = n_cols * patch_size[1]
        else:
            orig_w = n_cols * (patch_size[1] - stride[1]) + stride[1]

        if method == "avg":
            reconstruded = PatchReconstructor.combine_patches_avg(output, n_classes, orig_h, orig_w, patch_size, stride)
        elif method == "max":
            reconstruded = PatchReconstructor.combine_patches_max(output, n_classes, orig_h, orig_w, patch_size, stride)
        else:
            raise ValueError("Error: The combination method must be 'avg' or 'max'.")

        reconstruded = reconstruded[:, padding['top']:orig_h-padding['bottom'], padding['left']:orig_w-padding['right']]

        return reconstruded

    @staticmethod
    def combine_patches_avg(output: torch.Tensor, n_classes: int, original_heigh: int, original_width: int, patch_size: tuple = (256, 256), stride: tuple = (128, 128)) -> torch.Tensor:
        reconstructed = np.zeros((n_classes, original_heigh, original_width), dtype=np.float32)
        count_map = np.zeros((original_heigh, original_width), dtype=np.float32)

        idx = 0
        for y in range(0, original_heigh - patch_size[0] + 1, stride[0]):
            for x in range(0, original_width - patch_size[1] + 1, stride[1]):
                output_np = output[idx].detach().cpu().numpy()  # Transform to numpy
                reconstructed[:, y:y+patch_size[0], x:x+patch_size[1]] += output_np
                count_map[y:y+patch_size[0], x:x+patch_size[1]] += 1
                idx += 1

        reconstructed /= count_map
        return torch.Tensor(reconstructed)
    
    @staticmethod
    def combine_patches_max(output: torch.Tensor, n_classes: int, original_heigh: int, original_width: int, patch_size: tuple = (256, 256), stride: tuple = (128, 128)) -> torch.Tensor:
        reconstructed = np.zeros((n_classes, original_heigh, original_width), dtype=np.float32)
        
        idx = 0
        for y in range(0, original_heigh - patch_size[0] + 1, stride[0]):
            for x in range(0, original_width - patch_size[1] + 1, stride[1]):
                output_np = output[idx].detach().cpu().numpy()  # Transform to numpy
                reconstructed[:, y:y+patch_size[0], x:x+patch_size[1]] = np.maximum(reconstructed[:, y:y+patch_size[0], x:x+patch_size[1]], output_np)
                idx += 1

        return torch.Tensor(reconstructed)