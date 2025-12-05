import torch
import numpy as np
from pathlib import Path
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm

from image_datasets import get_image_dataset
from flow_matching.models import UNetModel
from flow_matching.solver import ModelWrapper, ODESolver


@torch.no_grad()
def generate_samples(flow, input_shape, n=150, device="cuda"):
    x = torch.randn((n, *input_shape), device=device)
    t = torch.linspace(0, 1, 101, device=device)

    class Wrapper(ModelWrapper):
        def forward(self, x, t, **extras):
            return self.model(x=x, t=t, **extras)

    solver = ODESolver(Wrapper(flow))
    samples = solver.sample(
        x_init=x,
        step_size=0.05,
        method="midpoint",
        time_grid=t,
        return_intermediates=False,
    ).cpu()

    return samples


def compute_distances(gen, real):
    gen_flat = gen.view(gen.size(0), -1)
    real_flat = real.view(real.size(0), -1)
    dists = torch.cdist(gen_flat, real_flat)  # (N_gen × N_real)
    return dists


def make_memorisation_grid(gen, real, dists, save_path):
    # Nearest & farthest indices
    nn_idx = dists.argmin(dim=1)
    max_idx = dists.argmax(dim=1)

    # Sort samples globally
    sorted_closest = torch.argsort(dists.min(dim=1).values)[:5]
    sorted_farthest = torch.argsort(dists.min(dim=1).values, descending=True)[:5]

    closest_gen = gen[sorted_closest]
    closest_real = real[nn_idx[sorted_closest]]

    farthest_gen = gen[sorted_farthest]
    farthest_real = real[nn_idx[sorted_farthest]]

    # Row order:
    # 1) 5 closest generated
    # 2) 5 matching nearest real
    # 3) 5 farthest generated
    # 4) 5 matching nearest real
    final = torch.cat([closest_gen,
                       closest_real,
                       farthest_gen,
                       farthest_real], dim=0)

    grid = make_grid(final, nrow=5, normalize=True)
    save_image(grid, save_path)
    print(f"Saved memorisation grid to {save_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load dataset ---
    dataset = get_image_dataset("boom_clay", root="/path/to/crops", train=True,
                                transform=None)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    real_imgs, _ = next(iter(loader))
    real_imgs = real_imgs.to(device)

    # --- Load trained model ---
    ckpt = "/path/to/outputs/cfm/boom_clay/ckpt.pth"
    input_shape = real_imgs[0].shape

    flow = UNetModel(input_shape, num_channels=64, num_res_blocks=2,
                     num_classes=1, class_cond=True).to(device)
    flow.load_state_dict(torch.load(ckpt, map_location=device))
    flow.eval()

    # --- Generate 150 samples ---
    gen_imgs = generate_samples(flow, input_shape, n=150, device=device)

    # --- Compute distance matrix (150 × N_train) ---
    dists = compute_distances(gen_imgs, real_imgs)

    # --- Make visual grid ---
    save_path = Path("memorisation_grid.png")
    make_memorisation_grid(gen_imgs, real_imgs, dists, save_path)


if __name__ == "__main__":
    main()
