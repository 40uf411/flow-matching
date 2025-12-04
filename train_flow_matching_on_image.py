from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm as std_tqdm
from transformers import HfArgumentParser

from flow_matching.datasets.image_datasets import (
    get_image_dataset,
    get_test_transform,
    get_train_transform,
)
from flow_matching.models import UNetModel
from flow_matching.sampler import PathSampler
from flow_matching.solver import ModelWrapper, ODESolver
from flow_matching.utils import model_size_summary, set_seed

tqdm = partial(std_tqdm, dynamic_ncols=True)


def resolve_image_size(dataset: str, override: int | None = None) -> int | None:
    """Return the target square image size compatible with UNet."""
    if override is not None:
        return override

    default_sizes = {
        "mnist": 28,
        "fashion_mnist": 28,
        "cifar10": 32,
        "celeba": 64,
        "boom_clay": 256,
    }
    return default_sizes.get(dataset)


@dataclass
class ScriptArguments:
    do_train: bool = True
    do_sample: bool = True
    dataset: str = "boom_clay"
    batch_size: int = 128
    n_epochs: int = 100
    learning_rate: float = 1e-3
    sigma_min: float = 0.0
    seed: int = 42
    output_dir: str = "outputs"
    horizontal_flip: bool = False
    image_size: int | None = None


def train(args: ScriptArguments):
    """Train the flow matching model on the given dataset."""

    output_dir = Path(args.output_dir) / "cfm" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"Using device: {device}")

    target_image_size = resolve_image_size(args.dataset, args.image_size)

    # Load the dataset
    dataset = get_image_dataset(
        args.dataset,
        train=True,
        transform=get_train_transform(
            horizontal_flip=args.horizontal_flip, image_size=target_image_size
        ),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Loaded {args.dataset} dataset with {len(dataset):,} samples")

    num_classes = len(dataset.classes)
    input_shape = dataset[0][0].size()
    print(f"{input_shape=}, {num_classes=}")

    # Load the UNet model with class conditioning for flow matching
    flow = UNetModel(
        input_shape,
        num_channels=64,
        num_res_blocks=2,
        num_classes=num_classes,
        class_cond=True,
    ).to(device)
    path_sampler = PathSampler(sigma_min=args.sigma_min)

    # Load the optimizer
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.learning_rate)
    scaler = GradScaler(enabled=device.type == "cuda")
    print("GradScaler enabled:", scaler._enabled)
    model_size_summary(flow)

    for epoch in range(args.n_epochs):
        flow.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:2d}/{args.n_epochs}")

        for x_1, y in pbar:
            x_1, y = x_1.to(device), y.to(device)

            # Compute the probability path samples
            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.size(0), device=device, dtype=x_1.dtype)
            x_t, dx_t = path_sampler.sample(x_0, x_1, t)

            flow.zero_grad(set_to_none=True)

            # Compute the conditional flow matching loss with class conditioning
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                vf_t = flow(t=t, x=x_t, y=y)
                loss = F.mse_loss(vf_t, dx_t)

            # Gradient scaling and backprop
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"loss": loss.item()})

    torch.save(flow.state_dict(), output_dir / "ckpt.pth")
    print(f"Final checkpoint saved to {output_dir / 'ckpt.pth'}")


def generate_samples_and_save_animation(args: ScriptArguments):
    """Generate samples following the flow and save the animation."""

    output_dir = Path(args.output_dir) / "cfm" / args.dataset
    assert output_dir.is_dir(), f"Output directory {output_dir} does not exist"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"Using device: {device}")

    target_image_size = resolve_image_size(args.dataset, args.image_size)

    # Load the dataset
    dataset = get_image_dataset(
        args.dataset,
        train=False,
        transform=get_test_transform(image_size=target_image_size),
    )
    input_shape = dataset[0][0].size()
    num_classes = len(dataset.classes)

    # Load the flow model
    flow = UNetModel(
        input_shape,
        num_channels=64,
        num_res_blocks=2,
        num_classes=num_classes,
        class_cond=True,
    ).to(device)
    state_dict = torch.load(output_dir / "ckpt.pth", weights_only=True)
    flow.load_state_dict(state_dict)
    flow.eval()

    # Use ODE solver to sample trajectories
    class WrappedModel(ModelWrapper):
        def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
            return self.model(x=x, t=t, **extras)

    samples_per_class = 10
    sample_steps = 101
    time_steps = torch.linspace(0, 1, sample_steps).to(device)
    class_list = torch.arange(num_classes, device=device).repeat(samples_per_class)

    wrapped_model = WrappedModel(flow)
    step_size = 0.05
    x_init = torch.randn((class_list.size(0), *input_shape), dtype=torch.float32, device=device)
    solver = ODESolver(wrapped_model)
    sol = solver.sample(
        x_init=x_init,
        step_size=step_size,
        method="midpoint",
        time_grid=time_steps,
        return_intermediates=True,
        y=class_list,
    )
    sol = sol.detach().cpu()
    final_samples = sol[-1]

    save_image(final_samples, output_dir / "final_samples.png", nrow=num_classes, normalize=True)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    grid = make_grid(final_samples, nrow=num_classes, normalize=True)
    ax[0].imshow(grid.permute(1, 2, 0))
    ax[0].set_title("Final samples (t = 1.0)", fontsize=16)
    ax[0].axis("off")

    def update(frame: int):
        grid = make_grid(sol[frame], nrow=num_classes, normalize=True)
        ax[1].clear()
        ax[1].imshow(grid.permute(1, 2, 0))
        ax[1].set_title(f"t = {time_steps[frame].item():.2f}", fontsize=16)
        ax[1].axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05, wspace=0.1)
    ani = animation.FuncAnimation(fig, update, frames=sample_steps)
    ani.save(output_dir / "trajectory.gif", writer="pillow", fps=20)
    print(f"Generated trajectory saved to {output_dir / 'trajectory.gif'}")


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args, *_ = parser.parse_args_into_dataclasses()

    if script_args.do_train:
        train(script_args)

    if script_args.do_sample:
        generate_samples_and_save_animation(script_args)
