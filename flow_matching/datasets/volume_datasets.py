from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryNpyVolumeDataset(Dataset):
    """
    Dataset for loading binary 3D volumes stored as .npy files.

    The dataset expects one volume per .npy file. Each file should contain a 3D
    array shaped (D, H, W) with values in {0, 1}. The tensor returned is shaped
    (1, D, H, W) so it can be fed directly to a 3D UNet.
    """

    def __init__(self, root: str | Path, split: str = "train", transform: Callable | None = None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        split_dir = self.root / split
        search_dir = split_dir if split_dir.exists() else self.root
        self.files = sorted(search_dir.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy files found in {search_dir.resolve()}")

        self.classes = ["volume"]
        self.targets = [0] * len(self.files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D volume in {path}, got shape {arr.shape}")

        tensor = torch.as_tensor(arr, dtype=torch.float32)
        tensor = tensor.unsqueeze(0)  # (1, D, H, W)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, 0


def get_volume_dataset(
    root: str | Path,
    split: str = "train",
    transform: Callable | None = None,
) -> BinaryNpyVolumeDataset:
    """
    Convenience wrapper to construct the binary volume dataset.
    """

    return BinaryNpyVolumeDataset(root=root, split=split, transform=transform)


def get_volume_transform(
    normalize: bool = True,
    random_flip: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a simple transform pipeline for 3D volumes.

    Args:
        normalize: If True, map values from [0, 1] to [-1, 1] to match the UNet's
            expected input range.
        random_flip: If True, apply random flips along each spatial axis.
    """

    def _transform(volume: torch.Tensor) -> torch.Tensor:
        if random_flip:
            if torch.rand(1) < 0.5:
                volume = torch.flip(volume, dims=[1])  # depth
            if torch.rand(1) < 0.5:
                volume = torch.flip(volume, dims=[2])  # height
            if torch.rand(1) < 0.5:
                volume = torch.flip(volume, dims=[3])  # width

        if normalize:
            volume = volume * 2 - 1  # [0, 1] -> [-1, 1]

        return volume

    return _transform
