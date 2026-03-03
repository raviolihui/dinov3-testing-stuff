import os
from typing import Any, Callable, Optional

import torch
import rasterio
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode


class BigEarthNet12Band(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 224,
        transform: Optional[Callable[[torch.Tensor], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
        transforms: Optional[Callable[[torch.Tensor, Any], Any]] = None,
        band_masking_prob: float = 0.0,
    ):
        """
        BigEarthNet Sentinel-2 12-band dataset (pretraining, unlabeled).

        Expects directory structure like:
          <root>/<scene_id>/<patch_id>/<patch_id>_B01.tif, ..., <patch_id>_B12.tif

        Args:
          root: Root folder of BigEarthNet-S2.
          image_size: Spatial size to which each band is resized (H=W=image_size).
          transform: Transform applied to the stacked 12xHxW tensor; used by DINO pipeline.
          target_transform: Ignored (no labels), kept for API compatibility.
          transforms: Ignored; provided for torchvision compatibility.
          band_masking_prob: Probability of masking each non-RGB band (0.0 = disabled).
        """
        self.root = root
        self.image_size = image_size
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.band_masking_prob = band_masking_prob
        
        # Initialize band masking transform if enabled
        if self.band_masking_prob > 0:
            from ..band_masking import RandomBandMasking
            self.band_masking = RandomBandMasking(
                mask_prob=band_masking_prob,
                mask_value=0.0,
                rgb_indices=(0, 1, 2)  # B04 (Red), B03 (Green), B02 (Blue)
            )
            print(f"[BigEarthNet] Band masking enabled: {self.band_masking}")
        else:
            self.band_masking = None

        self.band_names = [
            'B04','B03','B02','B01','B05','B06',
            'B07','B08','B8A','B09','B11','B12'
        ]

        self.patch_dirs = []
        for scene in os.listdir(root):
            scene_path = os.path.join(root, scene)
            if not os.path.isdir(scene_path):
                continue
            for patch in os.listdir(scene_path):
                patch_path = os.path.join(scene_path, patch)
                if os.path.isdir(patch_path):
                    self.patch_dirs.append(patch_path)

        print(f"[BigEarthNet] {len(self.patch_dirs)} patches found")

    def __len__(self):
        return len(self.patch_dirs)

    def __getitem__(self, idx):
        patch_dir = self.patch_dirs[idx]
        patch_id = os.path.basename(patch_dir)

        bands = []
        for band in self.band_names:
            path = os.path.join(patch_dir, f"{patch_id}_{band}.tif")
            with rasterio.open(path) as src:
                x = torch.from_numpy(src.read(1)).float()
                x = x.unsqueeze(0)
                # If image_size is provided, resize to the requested spatial size.
                # If image_size is None, keep native band resolution (no resizing).
                if self.image_size is not None:
                    x = resize(
                        x,
                        [self.image_size, self.image_size],
                        interpolation=InterpolationMode.BILINEAR,
                    )
                bands.append(x)

        x = torch.cat(bands, dim=0)
        # Scale reflectance to roughly [0, 1]; values are typically in [0, 10000]
        
        x = x / 10000.0  # BigEarthNet standard scaling
        
        # Apply band masking BEFORE other transforms (so augmentations see masked bands)
        if self.band_masking is not None:
            x = self.band_masking(x)
        
        # Apply transform pipeline if provided (DINO augmentations expect a tensor image)
        if self.transforms is not None:
            x, _ = self.transforms(x, None)
        elif self.transform is not None:
            x = self.transform(x)

        # No labels for self-supervised pretraining; return placeholder None
        target = None
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)

        return x, target
