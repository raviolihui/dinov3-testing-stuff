# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from typing import Sequence, Optional

import torch
from torchvision.transforms import v2

logger = logging.getLogger("dinov3")


def make_interpolation_mode(mode_str: str) -> v2.InterpolationMode:
    return {mode.value: mode for mode in v2.InterpolationMode}[mode_str]


class GaussianBlur(v2.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = v2.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# ForestNet statistics (per-band) computed from the dataset.
#
# The available ForestNet bands are, in order:
#   02 - Blue, 03 - Green, 04 - Red, 05 - NIR, 06 - SWIR1, 07 - SWIR2
# with the following (mean, std) over 0-255 uint8 range:
#   Blue:  mean=72.852258, std=15.837173
#   Green: mean=83.677155, std=14.788813
#   Red:   mean=77.581810, std=16.100543
#   NIR:   mean=123.987442, std=16.352349
#   SWIR1: mean=91.536942, std=13.788274
#   SWIR2: mean=74.719202, std=12.691314
#
# We normalise by 255 to work on [0, 1] floats, matching the rest of the
# pipeline, and define the 6-band ForestNet mean/std accordingly. These are
# then expanded/mapped to 12 channels when constructing ForestNet 12-band
# inputs so that each DINO channel is normalised with band-appropriate
# statistics.
FORESTNET_6B_MEAN = (
    72.852258 / 255.0,  # Blue
    83.677155 / 255.0,  # Green
    77.58181 / 255.0,   # Red
    123.987442 / 255.0, # NIR
    91.536942 / 255.0,  # SWIR1
    74.719202 / 255.0,  # SWIR2
)

FORESTNET_6B_STD = (
    15.837172547567825 / 255.0,  # Blue
    14.788812599596188 / 255.0,  # Green
    16.100543441881086 / 255.0,  # Red
    16.35234883118129 / 255.0,   # NIR
    13.7882739778638 / 255.0,    # SWIR1
    12.69131413539181 / 255.0,   # SWIR2
)


CROP_DEFAULT_SIZE = 224
RESIZE_DEFAULT_SIZE = int(256 * CROP_DEFAULT_SIZE / 224)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Normalize:
    return v2.Normalize(mean=mean, std=std)


def make_base_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Normalize:
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            make_normalize_transform(mean=mean, std=std),
        ]
    )


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=v2.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [v2.ToImage(), v2.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(v2.RandomHorizontalFlip(hflip_prob))
    transforms_list.append(make_base_transform(mean, std))
    transform = v2.Compose(transforms_list)
    logger.info(f"Built classification train transform\n{transform}")
    return transform


def make_resize_transform(
    *,
    resize_size: int,
    resize_square: bool = False,
    resize_large_side: bool = False,  # Set the larger side to resize_size instead of the smaller
    interpolation: v2.InterpolationMode = v2.InterpolationMode.BICUBIC,
):
    assert not (resize_square and resize_large_side), "These two options can not be set together"
    if resize_square:
        logger.info("resizing image as a square")
        size = (resize_size, resize_size)
        transform = v2.Resize(size=size, interpolation=interpolation)
        return transform
    elif resize_large_side:
        logger.info("resizing based on large side")
        transform = v2.Resize(size=None, max_size=resize_size, interpolation=interpolation)
        return transform
    else:
        transform = v2.Resize(resize_size, interpolation=interpolation)
        return transform


# Derived from make_classification_eval_transform() with more control over resize and crop
def make_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    resize_square: bool = False,
    resize_large_side: bool = False,  # Set the larger side to resize_size instead of the smaller
    interpolation: v2.InterpolationMode = v2.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Compose:
    transforms_list = [v2.ToImage()]
    resize_transform = make_resize_transform(
        resize_size=resize_size,
        resize_square=resize_square,
        resize_large_side=resize_large_side,
        interpolation=interpolation,
    )
    transforms_list.append(resize_transform)
    if crop_size:
        transforms_list.append(v2.CenterCrop(crop_size))
    transforms_list.append(make_base_transform(mean, std))
    transform = v2.Compose(transforms_list)
    logger.info(f"Built eval transform\n{transform}")
    return transform


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=v2.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Compose:
    return make_eval_transform(
        resize_size=resize_size,
        crop_size=crop_size,
        interpolation=interpolation,
        mean=mean,
        std=std,
        resize_square=False,
        resize_large_side=False,
    )


def make_forestnet_12b_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=v2.InterpolationMode.BICUBIC,
) -> v2.Compose:
    """Eval transform for ForestNet 12-band inputs.

    The ForestNet dataset provides 6 bands (Blue, Green, Red, NIR, SWIR1,
    SWIR2). These are mapped to the 12 DINOv3 BigEarthNet channels as
    follows (DINO order in parentheses):

      0: B04 (Red)   <- ForestNet Red
      1: B03 (Green) <- ForestNet Green
      2: B02 (Blue)  <- ForestNet Blue
      3: B01 (Blue)  <- ForestNet Blue (reused)
      4: B05 (NIR)   <- ForestNet NIR
      5: B06 (SWIR1) <- ForestNet SWIR1
      6: B07 (SWIR2) <- ForestNet SWIR2
      7: B08 (NIR)   <- ForestNet NIR (reused)
      8: B8A (NIR)   <- ForestNet NIR (reused)
      9: B09 (NIR)   <- ForestNet NIR (reused)
     10: B11 (SWIR1) <- ForestNet SWIR1 (reused)
     11: B12 (SWIR2) <- ForestNet SWIR2 (reused)

    We construct a 12-element mean/std that mirrors this mapping from the
    6-band ForestNet statistics defined above. This way each logical DINO
    channel is normalised using the mean/std of the underlying ForestNet
    band it represents.
    """

    # Expand 6-band ForestNet statistics to 12-band DINO ordering using the
    # mapping described above.
    forestnet_12b_mean = (
        FORESTNET_6B_MEAN[2],  # B04 (Red)
        FORESTNET_6B_MEAN[1],  # B03 (Green)
        FORESTNET_6B_MEAN[0],  # B02 (Blue)
        FORESTNET_6B_MEAN[0],  # B01 (Blue) reused
        FORESTNET_6B_MEAN[3],  # B05 (NIR)
        FORESTNET_6B_MEAN[4],  # B06 (SWIR1)
        FORESTNET_6B_MEAN[5],  # B07 (SWIR2)
        FORESTNET_6B_MEAN[3],  # B08 (NIR) reused
        FORESTNET_6B_MEAN[3],  # B8A (NIR) reused
        FORESTNET_6B_MEAN[3],  # B09 (NIR) reused
        FORESTNET_6B_MEAN[4],  # B11 (SWIR1) reused
        FORESTNET_6B_MEAN[5],  # B12 (SWIR2) reused
    )

    forestnet_12b_std = (
        FORESTNET_6B_STD[2],  # B04 (Red)
        FORESTNET_6B_STD[1],  # B03 (Green)
        FORESTNET_6B_STD[0],  # B02 (Blue)
        FORESTNET_6B_STD[0],  # B01 (Blue) reused
        FORESTNET_6B_STD[3],  # B05 (NIR)
        FORESTNET_6B_STD[4],  # B06 (SWIR1)
        FORESTNET_6B_STD[5],  # B07 (SWIR2)
        FORESTNET_6B_STD[3],  # B08 (NIR) reused
        FORESTNET_6B_STD[3],  # B8A (NIR) reused
        FORESTNET_6B_STD[3],  # B09 (NIR) reused
        FORESTNET_6B_STD[4],  # B11 (SWIR1) reused
        FORESTNET_6B_STD[5],  # B12 (SWIR2) reused
    )

    if resize_size == 0 and crop_size == 0:
        transforms_list = [
         # Input is already a CHW tensor; just dtype + normalize.
            v2.ToDtype(torch.float32, scale=True),
            make_normalize_transform(mean=forestnet_12b_mean, std=forestnet_12b_std),
        ]
        transform = v2.Compose(transforms_list)
        logger.info(f"Built ForestNet 12B eval transform without resize/crop\n{transform}")
        return transform

    # Otherwise fall back to the regular eval transform
    return make_eval_transform(
        resize_size=resize_size,
        crop_size=crop_size,
        interpolation=interpolation,
        mean=forestnet_12b_mean,
        std=forestnet_12b_std,
        resize_square=False,
        resize_large_side=False,
    )


# Sentinel-2 (EuroSAT/BigEarthNet) 12-band stats
# Order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
# Values are raw digital numbers (approx 0-10000 reflectance).
# We divide by 10000.0 to match the [0, 1] scaling in dataset.
SENTINEL2_MEAN = (
    1353.72696296 / 10000.0, # B01
    1117.20222222 / 10000.0, # B02
    1041.8842963  / 10000.0, # B03
    946.554      / 10000.0, # B04
    964.42577778 / 10000.0, # B05
    948.77244444 / 10000.0, # B06
    1016.46296296 / 10000.0, # B07
    1152.38962963 / 10000.0, # B08
    1530.84888889 / 10000.0, # B8A
    1551.18666667 / 10000.0, # B09
    741.62577778 / 10000.0,  # B11
    513.00118519 / 10000.0,  # B12
)

SENTINEL2_STD = (
    65.47953569 / 10000.0, # B01
    154.01972765 / 10000.0, # B02
    187.99427042 / 10000.0, # B03
    278.5087595  / 10000.0, # B04
    228.1849881  / 10000.0, # B05
    226.77241031 / 10000.0, # B06
    238.25816738 / 10000.0, # B07
    272.711904   / 10000.0, # B08
    321.46911369 / 10000.0, # B8A
    348.57242083 / 10000.0, # B09
    303.85698305 / 10000.0, # B11
    369.80735893 / 10000.0, # B12
)

# ImageNet stats replicated 4 times (for 12 bands) to support min-max scaled inputs
SENTINEL2_MINMAX_MEAN =  IMAGENET_DEFAULT_MEAN * 4
SENTINEL2_MINMAX_STD = IMAGENET_DEFAULT_STD * 4

# No custom mean/std arguments needed - we force the ImageNet replicated stats
def make_sentinel2_12b_minmax_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=v2.InterpolationMode.BICUBIC,
    mean: Optional[Sequence[float]] = SENTINEL2_MINMAX_MEAN,
    std: Optional[Sequence[float]] = SENTINEL2_MINMAX_STD,
) -> v2.Compose:
    """Eval transform for Sentinel-2 12-band inputs that have been Min-Max scaled to [0, 255].
    
    This matches the 'RGB' approach where we:
    1. Scale to [0, 255] (done in dataset loader via band_stats)
    2. Resize to 224 (done here)
    3. Normalize with ImageNet stats (done here, replicated for 12 bands) unless mean/std is None.
    """
    transforms_list = []
    
    # Dataset returns uint8 tensor (C, H, W).
    # Resize handles this fine.
    resize_transform = make_resize_transform(
        resize_size=resize_size,
        resize_square=False,
        resize_large_side=False,
        interpolation=interpolation,
    )
    if not isinstance(resize_transform, v2.Identity):
        transforms_list.append(resize_transform)

    if crop_size:
        transforms_list.append(v2.CenterCrop(crop_size))
        
    if mean is None or std is None:
        # Just scale to [0,1]
        transforms_list.append(v2.ToDtype(torch.float32, scale=True))
    else:
        transforms_list.append(make_base_transform(
            mean=mean, 
            std=std
        ))
    
    transform = v2.Compose(transforms_list)
    logger.info(f"Built Sentinel-2 12B MinMax eval transform\n{transform}")
    return transform


def voc2007_classification_target_transform(label, n_categories=20):
    one_hot = torch.zeros(n_categories, dtype=int)
    for instance in label.instances:
        one_hot[instance.category_id] = True
    return one_hot


def imaterialist_classification_target_transform(label, n_categories=294):
    one_hot = torch.zeros(n_categories, dtype=int)
    one_hot[label.attributes] = True
    return one_hot


def get_target_transform(dataset_str):
    if "VOC2007" in dataset_str:
        return voc2007_classification_target_transform
    elif "IMaterialist" in dataset_str:
        return imaterialist_classification_target_transform
    return None
