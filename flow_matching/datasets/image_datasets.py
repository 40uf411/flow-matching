import random
from collections.abc import Callable
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST, CelebA, FashionMNIST
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Compose, Normalize, RandomHorizontalFlip, Resize, ToDtype, ToImage
from torchvision.utils import save_image


class TiffRandomCropDataset(Dataset):
    """Single-class dataset that yields random 500x500 crops from .tiff files."""

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")

        self.samples = sorted(
            [p for p in self.root.iterdir() if p.suffix.lower() in (".png", ".jpg")]
        )
        if not self.samples:
            raise ValueError(f"No PNG/JPG crop files found in {self.root}")

        self.transform = transform
        self.classes = ["boom_clay"]
        self.targets = [0] * len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
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
        crops_root = Path(root) if root else Path("/home/ucl/elen/aaouf/boom_clay_crops/")
        return TiffRandomCropDataset(crops_root, transform=transform)
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


def save_boom_clay_real_grid(
    output_path: str | Path,
    root: str | Path | None = None,
    grid_size: int = 25,
    image_size: int | None = None,
    crop_size: int = 500,
) -> Path:
    """
    Save a grid of real boom_clay crops for visual comparison.

    Args:
        output_path: Where to save the PNG.
        root: Optional dataset root overriding the default boom_clay path.
        grid_size: Grid dimension (grid_size x grid_size images).
        image_size: Optional resize (matches model input if provided).
        crop_size: Random crop size before any resize.
    """

    n_samples = grid_size * grid_size
    dataset = get_image_dataset(
        "boom_clay",
        root=root,
        train=True,
        transform=get_test_transform(image_size=image_size),
        synthetic_length=n_samples,
        crop_size=crop_size,
    )

    images = []
    for i in range(n_samples):
        img, _ = dataset[i]
        images.append(img)

    batch = torch.stack(images, dim=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(batch, output_path, nrow=grid_size, normalize=True)
    return output_path


if __name__ == "__main__":
    # Convenience entrypoint to dump a real-image grid for comparison with generated samples.
    default_output = Path(__file__).parents[2] / "outputs" / "cfm" / "boom_clay" / "real_grid.png"
    saved_path = save_boom_clay_real_grid(
        output_path=default_output,
        root=None,
        grid_size=25,
        image_size=256,  # match the default boom_clay training size
        crop_size=500,
    )
    print(f"Saved boom_clay real grid to {saved_path}")
