from dataclasses import dataclass
from functools import partial
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm as std_tqdm
from transformers import HfArgumentParser

from flow_matching.datasets.volume_datasets import get_volume_dataset, get_volume_transform
from flow_matching.models import UNetModel
from flow_matching.sampler import PathSampler
from flow_matching.solver import ModelWrapper, ODESolver
from flow_matching.utils import model_size_summary, set_seed

tqdm = partial(std_tqdm, dynamic_ncols=True)


def first_slice(volumes: Tensor) -> Tensor:
    """Return the first axial slice (B, C, H, W) from a batch of volumes (B, C, D, H, W)."""
    if volumes.dim() != 5:
        raise ValueError(f"Expected volumes shaped (B, C, D, H, W), got {volumes.shape}")
    return volumes[:, :, 0, :, :]


def grid_rows(n: int) -> int:
    return max(1, int(math.sqrt(n)))


def sample_volumes(
    flow: UNetModel,
    input_shape: torch.Size,
    device: torch.device,
    num_classes: int,
    args: "VolumeScriptArguments",
):
    """Sample a batch of volumes and return the final timestep tensor."""

    class WrappedModel(ModelWrapper):
        def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:  # type: ignore[override]
            return self.model(x=x, t=t, **extras)

    time_steps = torch.linspace(0, 1, args.sample_steps, device=device)
    x_init = torch.randn((args.samples_to_save, *input_shape), device=device, dtype=torch.float32)

    sample_kwargs = {}
    if args.class_cond:
        repeats = (args.samples_to_save + num_classes - 1) // num_classes
        class_list = torch.arange(num_classes, device=device).repeat(repeats)[: args.samples_to_save]
        sample_kwargs["y"] = class_list

    solver = ODESolver(WrappedModel(flow))
    final_volumes = solver.sample(
        x_init=x_init,
        step_size=args.step_size,
        method="midpoint",
        time_grid=time_steps,
        return_intermediates=False,
        **sample_kwargs,
    )
    return final_volumes


@dataclass
class VolumeScriptArguments:
    data_root: str = "data/volumes"
    dataset_name: str = "binary_volumes"
    train_split: str = "train"
    eval_split: str = "train"
    do_train: bool = True
    do_sample: bool = True
    batch_size: int = 2
    n_epochs: int = 50
    learning_rate: float = 1e-4
    sigma_min: float = 0.0
    seed: int = 42
    output_dir: str = "outputs"
    random_flip: bool = False
    normalize: bool = True
    base_channels: int = 32
    num_res_blocks: int = 2
    class_cond: bool = False
    num_workers: int = 4
    sample_every: int = 5
    samples_to_save: int = 8
    sample_steps: int = 101
    step_size: float = 0.05


def train(args: VolumeScriptArguments):
    output_dir = Path(args.output_dir) / "cfm_3d" / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"Using device: {device}")

    # Dataset and loader
    transform = get_volume_transform(normalize=args.normalize, random_flip=args.random_flip)
    dataset = get_volume_dataset(root=args.data_root, split=args.train_split, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    print(f"Loaded {len(dataset):,} volumes from {args.train_split} split")

    num_classes = len(dataset.classes)
    input_shape = dataset[0][0].size()
    print(f"{input_shape=}, {num_classes=}")

    flow = UNetModel(
        input_shape,
        num_channels=args.base_channels,
        num_res_blocks=args.num_res_blocks,
        num_classes=num_classes if args.class_cond else None,
        class_cond=args.class_cond,
    ).to(device)
    path_sampler = PathSampler(sigma_min=args.sigma_min)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.learning_rate)
    scaler = GradScaler(enabled=device.type == "cuda")
    model_size_summary(flow)

    def save_sample_grid(epoch_num: int) -> None:
        flow.eval()
        with torch.no_grad():
            final_volumes = sample_volumes(flow, input_shape, device, num_classes, args)
        final_volumes = final_volumes.detach().cpu()
        torch.save(final_volumes, output_dir / f"sample_volumes_epoch_{epoch_num:04d}.pt")

        slices = first_slice(final_volumes)
        save_image(
            slices,
            output_dir / f"first_slice_epoch_{epoch_num:04d}.png",
            nrow=grid_rows(args.samples_to_save),
            normalize=True,
        )
        print(f"Saved sample slices for epoch {epoch_num}")

    for epoch in range(args.n_epochs):
        flow.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:2d}/{args.n_epochs}")

        for volumes, labels in pbar:
            volumes = volumes.to(device)
            labels = labels.to(device)

            x_0 = torch.randn_like(volumes)
            t = torch.rand(volumes.size(0), device=device, dtype=volumes.dtype)
            x_t, dx_t = path_sampler.sample(x_0, volumes, t)

            flow.zero_grad(set_to_none=True)
            amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                vf_t = flow(t=t, x=x_t, y=labels if args.class_cond else None)
                loss = F.mse_loss(vf_t, dx_t)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"loss": loss.item()})

        if (epoch + 1) % args.sample_every == 0:
            save_sample_grid(epoch + 1)

    torch.save(flow.state_dict(), output_dir / "ckpt.pth")
    print(f"Final checkpoint saved to {output_dir / 'ckpt.pth'}")


def generate_samples(args: VolumeScriptArguments):
    output_dir = Path(args.output_dir) / "cfm_3d" / args.dataset_name
    ckpt_path = output_dir / "ckpt.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"Using device: {device}")

    dataset = get_volume_dataset(
        root=args.data_root,
        split=args.eval_split,
        transform=get_volume_transform(normalize=args.normalize, random_flip=False),
    )
    num_classes = len(dataset.classes)
    input_shape = dataset[0][0].size()

    flow = UNetModel(
        input_shape,
        num_channels=args.base_channels,
        num_res_blocks=args.num_res_blocks,
        num_classes=num_classes if args.class_cond else None,
        class_cond=args.class_cond,
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    flow.load_state_dict(state_dict)
    flow.eval()

    with torch.no_grad():
        final_volumes = sample_volumes(flow, input_shape, device, num_classes, args)
    final_volumes = final_volumes.detach().cpu()
    torch.save(final_volumes, output_dir / "sample_volumes.pt")

    slices = first_slice(final_volumes)
    save_image(
        slices,
        output_dir / "first_slice_samples.png",
        nrow=grid_rows(args.samples_to_save),
        normalize=True,
    )
    print(f"Saved sampled volumes and first-slice grid to {output_dir}")


if __name__ == "__main__":
    parser = HfArgumentParser(VolumeScriptArguments)
    script_args, *_ = parser.parse_args_into_dataclasses()

    if script_args.do_train:
        train(script_args)

    if script_args.do_sample:
        generate_samples(script_args)
