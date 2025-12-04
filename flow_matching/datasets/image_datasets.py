import random
from collections.abc import Callable
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST, CelebA, FashionMNIST
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Compose, Normalize, RandomHorizontalFlip, Resize, ToDtype, ToImage


class TiffRandomCropDataset(Dataset):
    """Single-class dataset that yields random 500x500 crops from .tiff files."""

    def __init__(
        self,
        root: Path,
        transform: Callable | None = None,
        crop_size: int = 500,
        synthetic_length: int = 10_000,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"boom_clay dataset directory not found: {self.root}")

        self.samples = sorted(
            [p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() == ".tiff"]
        )
        if not self.samples:
            raise ValueError(f"No .tiff files found in boom_clay directory: {self.root}")

        self.transform = transform
        self.crop_size = crop_size
        self.synthetic_length = synthetic_length

        self.classes = ["boom_clay"]
        self.class_to_idx = {self.classes[0]: 0}
        self.targets = [0 for _ in range(self.synthetic_length)]

    def __len__(self) -> int:
        return self.synthetic_length

    def __getitem__(self, index: int):
        # Ignore index; pick a random file each time to amplify dataset size.
        path = random.choice(self.samples)
        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size
            if width < self.crop_size or height < self.crop_size:
                raise ValueError(
                    f"Image {path} is smaller than the requested crop size "
                    f"{self.crop_size}x{self.crop_size} (got {width}x{height})"
                )
            x0 = random.randint(0, width - self.crop_size)
            y0 = random.randint(0, height - self.crop_size)
            img = img.crop((x0, y0, x0 + self.crop_size, y0 + self.crop_size))

        if self.transform is not None:
            img = self.transform(img)

        return img, 0


def get_image_dataset(
    dataset_name: str,
    root: str | Path | None = None,
    train: bool = True,
    transform: Callable | None = None,
    synthetic_length: int | None = None,
    crop_size: int = 500,
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
        synthetic_len = synthetic_length if synthetic_length is not None else 10_000
        return TiffRandomCropDataset(
            boom_root, transform=transform, crop_size=crop_size, synthetic_length=synthetic_len
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


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
            ToImage(),  # convert to torchvision.tv_tensors.Image
            ToDtype(torch.float32, scale=True),  # scale to [0, 1]
        ]
    )
    if horizontal_flip:
        transform_list.append(RandomHorizontalFlip())
    if normalize:
        transform_list.append(Normalize((0.5,), (0.5,)))  # normalize to [-1, 1]
    return Compose(transform_list)


def get_test_transform(normalize: bool = True, image_size: int | None = None) -> Callable:
    transform_list = []
    if image_size is not None:
        transform_list.append(
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True)
        )
    transform_list.extend(
        [
            ToImage(),  # convert to torchvision.tv_tensors.Image
            ToDtype(torch.float32, scale=True),  # scale to [0, 1]
        ]
    )
    if normalize:
        transform_list.append(Normalize((0.5,), (0.5,)))  # normalize to [-1, 1]
    return Compose(transform_list)
