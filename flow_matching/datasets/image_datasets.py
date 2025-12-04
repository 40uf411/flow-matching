from collections.abc import Callable
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST, CelebA, FashionMNIST
from torchvision.transforms.v2 import Compose, Normalize, RandomHorizontalFlip, ToDtype, ToImage


class TiffFolderDataset(Dataset):
    """Single-class dataset that loads only .tiff files from a folder."""

    def __init__(self, root: Path, transform: Callable | None = None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"boom_clay dataset directory not found: {self.root}")

        self.samples = sorted(
            [p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() == ".tiff"]
        )
        if not self.samples:
            raise ValueError(f"No .tiff files found in boom_clay directory: {self.root}")

        self.transform = transform
        self.classes = ["boom_clay"]
        self.class_to_idx = {self.classes[0]: 0}
        self.targets = [0 for _ in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path = self.samples[index]
        with Image.open(path) as img:
            # Force RGB to keep channel count consistent for the model.
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, 0


def get_image_dataset(
    dataset_name: str,
    root: str | Path | None = None,
    train: bool = True,
    transform: Callable | None = None,
) -> Dataset:
    default_root = Path(__file__).parents[2] / "data"
    root_path = Path(root) if root is not None else default_root

    if dataset_name == "mnist":
        return MNIST(root_path, train, transform, download=True)
    elif dataset_name == "fashion_mnist":
        return FashionMNIST(root_path, train, transform, download=True)
    elif dataset_name == "cifar10":
        return CIFAR10(root_path, train, transform, download=True)
    elif dataset_name == "celeba":
        return CelebA(root_path, train, transform, download=True)  # gdown is required to download
    elif dataset_name == "boom_clay":
        boom_root = Path(root) if root is not None else Path("/home/ucl/elen/aaouf/lemmens_slices/")
        return TiffFolderDataset(boom_root, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_train_transform(horizontal_flip: bool = False, normalize: bool = True) -> Callable:
    transform_list = [
        ToImage(),  # convert to torchvision.tv_tensors.Image
        ToDtype(torch.float32, scale=True),  # scale to [0, 1]
    ]
    if horizontal_flip:
        transform_list.append(RandomHorizontalFlip())
    if normalize:
        transform_list.append(Normalize((0.5,), (0.5,)))  # normalize to [-1, 1]
    return Compose(transform_list)


def get_test_transform(normalize: bool = True) -> Callable:
    transform_list = [
        ToImage(),  # convert to torchvision.tv_tensors.Image
        ToDtype(torch.float32, scale=True),  # scale to [0, 1]
    ]
    if normalize:
        transform_list.append(Normalize((0.5,), (0.5,)))  # normalize to [-1, 1]
    return Compose(transform_list)
