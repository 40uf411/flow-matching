import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Module

from flow_matching.datasets import TOY_DATASETS
from flow_matching.solver import ModelWrapper, ODESolver


def sample(
    ode_model: ModelWrapper,
    source_samples: Tensor,
    sample_steps: int = 2,
    step_size: float = 0.05,
    method: str = "midpoint",
    return_intermediates: bool = False,
    **model_kwargs,
):
    device = next(ode_model.parameters()).device
    x_init = source_samples.to(device)
    time_grid = torch.linspace(0, 1, sample_steps).to(device)  # sample times
    solver = ODESolver(ode_model)
    samples = solver.sample(
        x_init=x_init,
        step_size=step_size,
        method=method,
        time_grid=time_grid,
        return_intermediates=return_intermediates,
        **model_kwargs,
    )
    return samples


class Swish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class Mlp(Module):
    def __init__(self, dim: int = 2, time_dim: int = 1, h: int = 64) -> None:
        super().__init__()
        self.input_dim = dim
        self.time_dim = time_dim
        self.hidden_dim = h
        self.layers = nn.Sequential(
            nn.Linear(dim + time_dim, h),
            Swish(),
            nn.Linear(h, h),
            Swish(),
            nn.Linear(h, h),
            Swish(),
            nn.Linear(h, dim),
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        size = x_t.size()
        x_t = x_t.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()
        t = t.reshape(-1, 1).expand(x_t.size(0), 1)
        h = torch.cat([x_t, t], dim=1)
        output = self.layers(h)
        return output.reshape(*size)


class WrappedModel(ModelWrapper):
    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        return self.model(x_t=x, t=t, **extras)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="checkerboard")
    parser.add_argument("--sample-steps", type=int, default=101)
    parser.add_argument("--num-samples", type=int, default=500_000)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfm_path = f"{args.output_dir}/cfm/{args.dataset}/ckpt.pth"
    reflow_path = f"{args.output_dir}/reflow/{args.dataset}/ckpt.pth"

    cfm = Mlp(dim=2, time_dim=1, h=512)
    cfm.load_state_dict(torch.load(cfm_path, weights_only=True))
    cfm.to(device)
    cfm.eval()
    wrapped_cfm = WrappedModel(cfm)

    reflow = Mlp(dim=2, time_dim=1, h=512)
    reflow.load_state_dict(torch.load(reflow_path, weights_only=True))
    reflow.to(device)
    reflow.eval()
    wrapped_reflow = WrappedModel(reflow)

    dataset = TOY_DATASETS[args.dataset](device=device)

    x_init = torch.randn(args.num_samples, 2).to(device)
    samples_cfm = sample(wrapped_cfm, x_init, sample_steps=args.sample_steps, return_intermediates=True)
    samples_reflow = sample(wrapped_reflow, x_init, sample_steps=args.sample_steps, return_intermediates=True)

    samples_cfm = samples_cfm.detach().cpu().numpy()
    samples_reflow = samples_reflow.detach().cpu().numpy()

    # Create a grid for the density and vector field
    grid_size = 15
    x_range, y_range = dataset.get_square_range()
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    xv, yv = np.meshgrid(x, y)
    grid = np.stack([xv, yv], axis=-1).reshape(-1, 2)  # Shape: (grid_size^2, 2)

    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)
    time_steps = torch.linspace(0, 1, args.sample_steps).to(device)

    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    def update(frame):
        for ax in axes.flatten():
            ax.clear()

        # Current time step
        t = time_steps[frame]
        t_tensor = torch.full((grid_tensor.size(0), 1), t, device=device)

        # Plot CFM samples
        axes[0, 0].hist2d(
            samples_cfm[frame, :, 0],
            samples_cfm[frame, :, 1],
            bins=300,
            range=[x_range, y_range],
            cmap="viridis",
        )
        axes[0, 0].set_title(f"Flow Matching (t = {t.item():.2f})", fontsize=16)
        axes[0, 0].set_xlim(x_range)
        axes[0, 0].set_ylim(y_range)
        axes[0, 0].set_aspect("equal")
        axes[0, 0].axis("off")

        # Plot Reflow samples
        axes[0, 1].hist2d(
            samples_reflow[frame, :, 0],
            samples_reflow[frame, :, 1],
            bins=300,
            range=[x_range, y_range],
            cmap="viridis",
        )
        axes[0, 1].set_title(f"Reflow (2-Rectified Flow) (t = {t.item():.2f})", fontsize=16)
        axes[0, 1].set_xlim(x_range)
        axes[0, 1].set_ylim(y_range)
        axes[0, 1].set_aspect("equal")
        axes[0, 1].axis("off")

        # Plot CFM vector field
        vectors_cfm = wrapped_cfm(grid_tensor, t_tensor).detach().cpu().numpy()
        vectors_cfm = vectors_cfm.reshape(grid_size, grid_size, 2)
        magnitudes_cfm = np.linalg.norm(vectors_cfm, axis=2)
        axes[1, 0].quiver(
            xv,
            yv,
            vectors_cfm[:, :, 0],
            vectors_cfm[:, :, 1],
            magnitudes_cfm,
            angles="xy",
            scale_units="xy",
            scale=10.0,
            cmap=cm.coolwarm,
            alpha=0.8,
            width=0.01,
        )

        # axes[1, 0].set_title(f"CFM Vector Field (t = {t.item():.2f})", fontsize=16)
        axes[1, 0].set_xlim(x_range)
        axes[1, 0].set_ylim(y_range)
        axes[1, 0].set_aspect("equal")
        axes[1, 0].axis("off")

        # Plot Reflow vector field
        vectors_reflow = wrapped_reflow(grid_tensor, t_tensor).detach().cpu().numpy()
        vectors_reflow = vectors_reflow.reshape(grid_size, grid_size, 2)
        magnitudes_reflow = np.linalg.norm(vectors_reflow, axis=2)
        axes[1, 1].quiver(
            xv,
            yv,
            vectors_reflow[:, :, 0],
            vectors_reflow[:, :, 1],
            magnitudes_reflow,
            angles="xy",
            scale_units="xy",
            scale=10.0,
            cmap=cm.coolwarm,
            alpha=0.8,
            width=0.01,
        )
        # axes[1, 1].set_title(f"Reflow Vector Field (t = {t.item():.2f})", fontsize=16)
        axes[1, 1].set_xlim(x_range)
        axes[1, 1].set_ylim(y_range)
        axes[1, 1].set_aspect("equal")
        axes[1, 1].axis("off")

    # Adjust layout to reduce white space
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.1, hspace=0.05)

    ani = animation.FuncAnimation(fig, update, frames=args.sample_steps)

    print("Saving animation...")
    folder = Path(args.output_dir) / "comparisons"
    folder.mkdir(parents=True, exist_ok=True)
    ani.save(folder / f"cfm_reflow_{args.dataset}.gif", writer="pillow", fps=20)


if __name__ == "__main__":
    main()
