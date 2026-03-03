# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar

import torch
from torch.utils.data import Sampler

from .datasets import ADE20K, CocoCaptions, ImageNet, ImageNet22k, BigEarthNet12Band, GeoBenchCls12
from .datasets import CoreS2L2A
from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler

logger = logging.getLogger("dinov3")


class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


def _make_bool_str(b: bool) -> str:
    return "yes" if b else "no"


def _make_sample_transform(
    image_transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    def transform(sample):
        image, target = sample
        if image_transform is not None:
            image = image_transform(image)
        if target_transform is not None:
            target = target_transform(target)
        return image, target

    return transform


def _parse_dataset_str(dataset_str: str):
    tokens = dataset_str.split(":")

    name = tokens[0]
    kwargs = {}

    for token in tokens[1:]:
        parts = token.split("=", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid token '{token}' in dataset string '{dataset_str}'. Expected 'key=value'.")
        key, value = parts
        if key not in ("root", "extra", "split", "partition_name", "band_masking_prob"):
            raise ValueError(f"Invalid key '{key}' in dataset string '{dataset_str}'. Allowed keys: root, extra, split, partition_name, band_masking_prob.")
        kwargs[key] = value

    if name == "ImageNet":
        class_ = ImageNet
        if "split" in kwargs:
            kwargs["split"] = ImageNet.Split[kwargs["split"]]
    elif name == "ImageNet22k":
        class_ = ImageNet22k
    elif name == "ADE20K":
        class_ = ADE20K
        if "split" in kwargs:
            kwargs["split"] = ADE20K.Split[kwargs["split"]]
    elif name == "CocoCaptions":
        class_ = CocoCaptions
        if "split" in kwargs:
            kwargs["split"] = CocoCaptions.Split[kwargs["split"]]
    elif name == "BigEarthNet12Band":
        # BigEarthNet 12-band Sentinel-2 patches
        # Expected kwargs: root=<path to BigEarthNet-S2 root>, band_masking_prob=<float>
        class_ = BigEarthNet12Band
        if "band_masking_prob" in kwargs:
            kwargs["band_masking_prob"] = float(kwargs["band_masking_prob"])
    elif name == "GeoBenchCls12":
        # GeoBench classification adapter for 12-band inputs (e.g. m-forestnet)
        # Expected kwargs: root=<geobench classification task root>, split=TRAIN|VAL|TEST
        class_ = GeoBenchCls12
        if "split" in kwargs:
            kwargs["split"] = GeoBenchCls12.Split[kwargs["split"]]
    elif name == "GeoBenchCls":
        # Check if the root path indicates a 12-band dataset that should use the 12-band adapter
        # Heuristic: if 'm-eurosat' or 'm-bigearthnet' is in root path, switch to GeoBenchCls12
        # This allows users to keep using "GeoBenchCls" string in config but get correct 12-band handling.
        root_path = kwargs.get("root", "")
        if "m-eurosat" in root_path or "m-bigearthnet" in root_path or "m-forestnet" in root_path:
             class_ = GeoBenchCls12
             if "split" in kwargs:
                 # Map split string to GeoBenchCls12 enum if slightly different (usually same)
                 kwargs["split"] = GeoBenchCls12.Split[kwargs["split"]]
        else:
             from .geobench import GeoBenchCls
             class_ = GeoBenchCls
             if "split" in kwargs:
                 kwargs["split"] = GeoBenchCls.Split[kwargs["split"]]
        
        if "extra" in kwargs:
            extra = kwargs.pop("extra")
            if extra:
                kwargs["band_names"] = tuple(extra.split(","))
    else:
        # Support Core-S2L2A dataset string: CoreS2L2A:root=...:band_masking_prob=0.3
        if name == "CoreS2L2A":
            class_ = CoreS2L2A
            if "band_masking_prob" in kwargs:
                kwargs["band_masking_prob"] = float(kwargs["band_masking_prob"])
        else:
            raise ValueError(f'Unsupported dataset "{name}"')

    return class_, kwargs


def make_dataset(
    *,
    dataset_str: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    transforms: Optional[Callable] = None,
):
    """
    Creates a dataset with the specified parameters.

    Args:
        dataset_str: A dataset string description (e.g. ImageNet:split=TRAIN).
        transform: A transform to apply to images.
        target_transform: A transform to apply to targets.
        transforms: A transform to apply to both images and targets.

    Returns:
        The created dataset.
    """
    logger.info(f'using dataset: "{dataset_str}"')

    class_, kwargs = _parse_dataset_str(dataset_str)
    dataset = class_(transform=transform, target_transform=target_transform, transforms=transforms, **kwargs)

    logger.info(f"# of dataset samples: {len(dataset):,d}")

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        dataset.transform = transform
    if not hasattr(dataset, "target_transform"):
        dataset.target_transform = target_transform
    if not hasattr(dataset, "transforms"):
        dataset.transforms = transforms

    return dataset


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
    worker_init_fn: Optional[Callable[[List[T]], Any]] = None,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, SHARDED_INFINITE_NEW, DISTRIBUTED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        sampler_advance: How many samples to skip (when applicable).
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
    """

    sampler = _make_sampler(
        dataset=dataset,
        type=sampler_type,
        shuffle=shuffle,
        seed=seed,
        size=sampler_size,
        advance=sampler_advance,
    )

    logger.info("using PyTorch data loader")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader
