from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.utils import save_image


class NpyVolumeDataset(Dataset):
    """
    Single-class dataset that loads 3D volumes from .npy files.

    Each file must contain:
      - (D,H,W)  -> returned as (1,D,H,W)
      - OR (C,D,H,W) -> returned as is

    Returns: (tensor, 0)
    """

    def __init__(self, root: Path, transform: Callable | None = None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")

        self.samples = sorted([p for p in self.root.iterdir() if p.suffix.lower() == ".npy"])
        if not self.samples:
            raise ValueError(f"No .npy files found in {self.root}")

        self.transform = transform
        self.classes = [self.root.name]
        self.targets = [0] * len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        p = self.samples[idx]
        arr = np.load(p)

        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Loaded object is not a numpy array: {type(arr)} from {p}")

        x = torch.from_numpy(arr)

        # Accept (D,H,W) or (C,D,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1,D,H,W)
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(f"Expected array of shape (D,H,W) or (C,D,H,W), got {tuple(x.shape)} from {p}")

        # Always float32 for the model
        x = x.to(torch.float32)

        if self.transform is not None:
            x = self.transform(x)

        return x, 0


def _normalize_to_minus1_plus1(x: torch.Tensor) -> torch.Tensor:
    """
    Robust normalization:
    - if max > 1.5 assume [0,255] and scale to [0,1]
    - else assume already in [0,1] or {0,1}
    Then map to [-1,1].
    """
    with torch.no_grad():
        mx = x.max()
        if mx > 1.5:
            x = x / 255.0
    return x * 2.0 - 1.0


def get_image_dataset(
    dataset_name: str,
    root: str | Path | None = None,
    train: bool = True,
    transform: Callable | None = None,
    synthetic_length: int | None = None,
    crop_size: int = 500,
) -> Dataset:
    if root is None:
        raise ValueError("For 3D datasets, `root` must point to the folder of .npy volumes.")

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"3D dataset folder not found: {root_path}")

    # IMPORTANT: no dataset_name switch/case here
    return NpyVolumeDataset(root_path, transform=transform)


def get_train_transform(
    horizontal_flip: bool = False,  # kept for signature compatibility
    normalize: bool = True,
    image_size: int | None = None,  # unused for fixed 64^3 volumes, kept for compatibility
) -> Callable:
    """
    Return a callable x -> x for volumes shaped (C,D,H,W).

    For 3D, we interpret `horizontal_flip=True` as random flips along D/H/W.
    """
    def _tf(x: torch.Tensor) -> torch.Tensor:
        if horizontal_flip:
            # Random flips across spatial axes
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[1])  # D
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[2])  # H
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[3])  # W

        if normalize:
            x = _normalize_to_minus1_plus1(x)

        return x

    return _tf


def get_test_transform(normalize: bool = True, image_size: int | None = None) -> Callable:
    def _tf(x: torch.Tensor) -> torch.Tensor:
        if normalize:
            x = _normalize_to_minus1_plus1(x)
        return x

    return _tf


def save_real_grid_mid_slices(
    output_path: str | Path,
    root: str | Path,
    dataset_name: str = "volumes",
    grid_size: int = 8,
) -> Path:
    """
    3D equivalent of saving a "real grid":
    takes mid-slice at D//2 from each volume and saves a 2D grid image.
    """
    n_samples = grid_size * grid_size
    ds = get_image_dataset(dataset_name, root=root, train=True, transform=get_test_transform())

    # sample first n_samples (or wrap-around if dataset smaller)
    slices = []
    for i in range(n_samples):
        vol, _ = ds[i % len(ds)]  # (C,D,H,W)
        dmid = vol.shape[1] // 2
        sl = vol[:, dmid, :, :]  # (C,H,W)
        slices.append(sl)

    batch = torch.stack(slices, dim=0)  # (B,C,H,W)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(batch, output_path, nrow=grid_size, normalize=True)
    return output_path
