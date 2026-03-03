# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Any, Tuple

from torchvision.datasets import VisionDataset

from .decoders import Decoder, ImageDataDecoder, TargetDecoder


class ExtendedVisionDataset(VisionDataset):
    def __init__(
        self,
        image_decoder: Decoder = ImageDataDecoder,
        target_decoder: Decoder = TargetDecoder,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.image_decoder = image_decoder
        self.target_decoder = target_decoder

    def get_image_data(self, index: int):
        """Return raw image data for a sample.

        Subclasses may return either encoded image bytes (e.g. PNG/JPEG)
        or a tensor/NumPy array. The default decoders only handle bytes,
        so non-byte types should be passed directly to the transform
        pipeline.
        """
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        # If the subclass returns encoded bytes, use the configured decoder.
        if isinstance(image_data, (bytes, bytearray, memoryview)):
            image = self.image_decoder(image_data).decode()
        else:
            # For tensor / NumPy array data, pass through as-is and let
            # the transforms handle it.
            image = image_data

        target = self.get_target(index)
        target = self.target_decoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        raise NotImplementedError
