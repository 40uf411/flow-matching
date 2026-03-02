# image_datasets_p3D.py
#
# Pseudo-3D dataloading for sparse-slice supervision from 2D images.
# Returns (Y, M, orient_id) where:
#   Y: [C, D, H, W] pseudo-volume with observed planes inserted, zeros elsewhere
#   M: [1, D, H, W] binary mask for observed voxels (1 where observed, 0 elsewhere)
#   orient_id: int in {0,1,2} corresponding to {"xy","xz","yz"}
#
# Notes:
# - Assumes volume_size == H == W == D (e.g., 64).
# - Expects per-orientation folders under root: root/xy, root/xz, root/yz.
# - The provided transform MUST output torch.Tensor [C, volume_size, volume_size].

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST, CelebA, FashionMNIST
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToDtype,
    ToImage,
)
from torchvision.utils import save_image


Orientation = Literal["xy", "xz", "yz"]


def _pick_slice_ids(
    D: int,
    k: int,
    min_gap: int,
    rng: random.Random,
) -> list[int]:
    """Pick k unique slice indices in [0, D-1] with a minimum spacing constraint."""
    if k <= 0:
        return []
    if k == 1:
        return [rng.randrange(D)]

    max_tries = 200
    for _ in range(max_tries):
        ids = sorted(rng.sample(range(D), k))
        ok = True
        for a, b in zip(ids[:-1], ids[1:]):
            if (b - a) < min_gap:
                ok = False
                break
        if ok:
            return ids

    # Fallback: unique without spacing constraint
    return sorted(rng.sample(range(D), k))


class Pseudo3DSlicesDataset(Dataset):
    """
    Builds a pseudo 3D volume by inserting K random 2D crops at random slice ids
    for ONE chosen orientation per returned sample.

    Returns:
        Y: [C, D, H, W]
        M: [1, D, H, W]
        orient_id: int {0:xy, 1:xz, 2:yz}
        (optional) debug dict if return_debug=True
    """

    ORIENTS: tuple[Orientation, ...] = ("xy", "xz", "yz")
    ORIENT_TO_ID = {"xy": 0, "xz": 1, "yz": 2}

    def __init__(
        self,
        root: Path,
        transform: Callable | None,
        volume_size: int = 64,
        k_slices: int = 3,
        min_gap: int = 2,
        choose_orientation: Literal["random", "cycle"] = "random",
        return_debug: bool = False,
        seed: int = 0,
        per_orient_roots: dict[Orientation, Path] | None = None,
        file_suffixes: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        self.root = Path(root)
        self.transform = transform

        self.D = int(volume_size)
        self.H = int(volume_size)
        self.W = int(volume_size)

        self.k_slices = int(k_slices)
        self.min_gap = int(min_gap)
        self.choose_orientation = choose_orientation
        self.return_debug = return_debug

        self.rng = random.Random(seed)
        self._cycle_idx = 0

        # Resolve per-orientation roots
        if per_orient_roots is None:
            per_orient_roots = {o: self.root / o for o in self.ORIENTS}
        self.per_orient_roots = {o: Path(p) for o, p in per_orient_roots.items()}

        # Collect samples per orientation
        self.samples: dict[Orientation, list[Path]] = {}
        for o in self.ORIENTS:
            r = self.per_orient_roots[o]
            if not r.is_dir():
                raise FileNotFoundError(f"Orientation folder not found: {r}")
            files = sorted([p for p in r.iterdir() if p.suffix.lower() in file_suffixes])
            if not files:
                raise ValueError(f"No image files found in {r} (suffixes={file_suffixes}).")
            self.samples[o] = files

        # Stable length: max count across orientations
        self._length = max(len(v) for v in self.samples.values())

        # Optional single-class metadata
        self.classes = ["banderabrown_bin"]
        self.targets = [0] * self._length

        # Basic sanity checks
        if self.transform is None:
            raise ValueError("transform must be provided and must return a torch.Tensor [C,H,W].")
        if self.D <= 0:
            raise ValueError("volume_size must be > 0.")
        if self.k_slices < 0:
            raise ValueError("k_slices must be >= 0.")
        if self.choose_orientation not in ("random", "cycle"):
            raise ValueError("choose_orientation must be 'random' or 'cycle'.")

    def __len__(self) -> int:
        return self._length

    def _choose_orientation(self, idx: int) -> Orientation:
        if self.choose_orientation == "cycle":
            o = self.ORIENTS[self._cycle_idx % len(self.ORIENTS)]
            self._cycle_idx += 1
            return o
        return self.ORIENTS[self.rng.randrange(len(self.ORIENTS))]

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        if not isinstance(x, torch.Tensor):
            raise TypeError("transform must return a torch.Tensor.")
        if x.ndim != 3:
            raise ValueError(f"Expected transformed image shape [C,H,W], got {tuple(x.shape)}")
        if x.shape[-2:] != (self.H, self.W):
            raise ValueError(
                f"Transform must output {(self.H, self.W)} spatial size, got {tuple(x.shape[-2:])}. "
                f"Add Resize((volume_size, volume_size)) to the transform."
            )
        return x

    def __getitem__(self, idx: int):
        o = self._choose_orientation(idx)
        orient_id = self.ORIENT_TO_ID[o]

        # Pick slice ids in [0..D-1] with optional spacing constraint
        slice_ids = _pick_slice_ids(self.D, self.k_slices, self.min_gap, self.rng)

        # Determine channels C from one transformed image
        probe_path = self.samples[o][idx % len(self.samples[o])]
        probe = self._load_image(probe_path)
        C = probe.shape[0]

        # Initialize pseudo volume + mask (zeros in unobserved voxels)
        Y = torch.zeros((C, self.D, self.H, self.W), dtype=probe.dtype)
        M = torch.zeros((1, self.D, self.H, self.W), dtype=probe.dtype)

        chosen_paths: list[Path] = []
        for s in slice_ids:
            p = self.samples[o][self.rng.randrange(len(self.samples[o]))]
            chosen_paths.append(p)
            crop = self._load_image(p)  # [C,H,W]

            # Canonical volume layout: [C, z, y, x] == [C, D, H, W]
            if o == "xy":
                # xy plane at z=s
                Y[:, s, :, :] = crop
                M[:, s, :, :] = 1.0
            elif o == "xz":
                # xz plane at y=s  => indices [z, x] stored at Y[:, z, s, x]
                # Since crop is [C,H,W] and H==D, interpret crop[:, z, x].
                Y[:, :, s, :] = crop
                M[:, :, s, :] = 1.0
            else:  # "yz"
                # yz plane at x=s  => indices [z, y] stored at Y[:, z, y, s]
                Y[:, :, :, s] = crop
                M[:, :, :, s] = 1.0

        if self.return_debug:
            debug = {
                "orientation": o,
                "slice_ids": slice_ids,
                "paths": [str(p) for p in chosen_paths],
            }
            return Y, M, orient_id, debug

        return Y, M, orient_id


def get_image_dataset(
    dataset_name: str,
    root: str | Path | None = None,
    train: bool = True,
    transform: Callable | None = None,
    synthetic_length: int | None = None,
    # Pseudo-3D args:
    volume_size: int = 64,
    k_slices: int = 3,
    min_gap: int = 2,
    choose_orientation: Literal["random", "cycle"] = "random",
    return_debug: bool = False,
) -> Dataset:
    """
    Dataset factory compatible with existing code patterns.

    For 'banderabrown_bin_p3d' it expects:
        root/xy/*.png, root/xz/*.png, root/yz/*.png (or jpg/jpeg)

    Returns datasets yielding:
        (image, target) for classic datasets
        (Y, M, orient_id) for pseudo-3D dataset
    """
    default_root = Path(__file__).parents[2] / "data"
    root_path = Path(root) if root is not None else default_root

    if dataset_name == "mnist":
        return MNIST(root_path, train, transform, download=True)
    if dataset_name == "fashion_mnist":
        return FashionMNIST(root_path, train, transform, download=True)
    if dataset_name == "cifar10":
        return CIFAR10(root_path, train, transform, download=True)
    if dataset_name == "celeba":
        return CelebA(root_path, train, transform, download=True)  # gdown may be required

    if dataset_name in ("banderabrown_bin_p3d", "banderabrown_p3d"):
        data_root = Path(root) if root else Path("/export/home/aaouf/workspace/2d_images/banderabrown_2D_dataset")
        # Use different seed for train vs test for deterministic-ish behavior if desired
        seed = 0 if train else 123
        ds = Pseudo3DSlicesDataset(
            root=data_root,
            transform=transform,
            volume_size=volume_size,
            k_slices=k_slices,
            min_gap=min_gap,
            choose_orientation=choose_orientation,
            return_debug=return_debug,
            seed=seed,
        )
        if synthetic_length is not None:
            return _LengthWrapper(ds, int(synthetic_length))
        return ds

    raise ValueError(f"Unknown dataset: {dataset_name}")


class _LengthWrapper(Dataset):
    """Wrap any dataset to present a synthetic __len__ while sampling items modulo base length."""

    def __init__(self, base: Dataset, length: int):
        self.base = base
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        return self.base[idx % len(self.base)]


def get_train_transform(
    horizontal_flip: bool = False,
    normalize: bool = True,
    image_size: int | None = None,
) -> Callable:
    transform_list = []
    if image_size is not None:
        transform_list.append(
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True)
        )
    transform_list.extend(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),  # [0,1]
        ]
    )
    if horizontal_flip:
        transform_list.append(RandomHorizontalFlip())
    if normalize:
        # NOTE: for RGB use 3 channels; if you want grayscale, change this.
        transform_list.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # [-1,1]
    return Compose(transform_list)


def get_test_transform(normalize: bool = True, image_size: int | None = None) -> Callable:
    transform_list = []
    if image_size is not None:
        transform_list.append(
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True)
        )
    transform_list.extend(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
        ]
    )
    if normalize:
        transform_list.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return Compose(transform_list)


def save_real_grid(
    output_path: str | Path,
    dataset_name: str,
    root: str | Path | None = None,
    grid_size: int = 25,
    image_size: int | None = None,
    volume_size: int = 64,
    k_slices: int = 3,
    min_gap: int = 2,
) -> Path:
    """
    Save a grid of real samples for visual comparison.

    - For classic 2D datasets: saves a grid of images.
    - For pseudo-3D dataset: saves a grid of randomly selected *observed xy slices* from Y
      (useful sanity check that your crops/masks look right).
    """
    n_samples = grid_size * grid_size
    transform = get_test_transform(image_size=image_size)

    dataset = get_image_dataset(
        dataset_name,
        root=root,
        train=True,
        transform=transform,
        synthetic_length=n_samples,
        volume_size=volume_size,
        k_slices=k_slices,
        min_gap=min_gap,
        choose_orientation="random",
        return_debug=False,
    )

    images = []
    for i in range(n_samples):
        sample = dataset[i]
        if dataset_name in ("banderabrown_bin_p3d", "banderabrown_p3d"):
            Y, M, orient_id = sample
            # Pick one observed slice index if any; otherwise use 0.
            # For grid visualization, take the central depth for readability.
            # Here we just take the first observed slice.
            msum = M.sum(dim=(0, 2, 3))  # [D]
            if torch.any(msum > 0):
                s = int(torch.nonzero(msum > 0, as_tuple=False)[0].item())
            else:
                s = 0
            # Visualize the slice depending on orientation:
            if orient_id == 0:  # xy
                img = Y[:, s, :, :]
            elif orient_id == 1:  # xz -> plane at y=s is Y[:, :, s, :]
                # pick a y with observations; fall back to center
                ysum = M.sum(dim=(0, 1, 3))  # [H] because mask is [1,D,H,W]
                y = int(torch.nonzero(ysum > 0, as_tuple=False)[0].item()) if torch.any(ysum > 0) else volume_size // 2
                img = Y[:, :, y, :]  # [C,D,W] but D==H so looks like [C,H,W]
            else:  # yz -> plane at x=s is Y[:, :, :, x]
                xsum = M.sum(dim=(0, 1, 2))  # [W]
                x = int(torch.nonzero(xsum > 0, as_tuple=False)[0].item()) if torch.any(xsum > 0) else volume_size // 2
                img = Y[:, :, :, x]  # [C,D,H]
            images.append(img)
        else:
            img, _ = sample
            images.append(img)

    batch = torch.stack(images, dim=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(batch, output_path, nrow=grid_size, normalize=True)
    return output_path


if __name__ == "__main__":
    # Example: pseudo-3D sanity grid (shows observed slices only)
    default_output = Path(__file__).parents[2] / "outputs" / "cfm" / "banderabrown_p3d" / "real_grid.png"
    saved_path = save_real_grid(
        output_path=default_output,
        dataset_name="banderabrown_bin_p3d",
        root="/export/home/aaouf/workspace/2d_images/banderabrown_2D_dataset",
        grid_size=10,
        image_size=64,     # must match volume_size for pseudo-3D
        volume_size=64,
        k_slices=3,
        min_gap=2,
    )
    print(f"Saved grid to {saved_path}")