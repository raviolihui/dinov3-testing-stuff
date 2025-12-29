import os
import torch
import rasterio
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode


class BigEarthNet12Band(Dataset):
    def __init__(self, root, image_size=224):
        self.root = root
        self.image_size = image_size

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
                x = resize(
                    x,
                    [self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR
                )
                bands.append(x)

        x = torch.cat(bands, dim=0)
        x = x / 10000.0  # BigEarthNet standard scaling
        return x
