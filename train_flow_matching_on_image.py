import os, tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
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

from flow_matching.eval import eval as run_eval, EvalConfig

# =========================
# Temp dirs / Comet dirs
# =========================
custom_temp_dir = os.path.abspath("./temp")
comet_cache_dir = os.path.abspath("./.comet-cache")
comet_offline_dir = os.path.abspath("./.comet-offline")
for directory in [custom_temp_dir, comet_cache_dir, comet_offline_dir]:
    os.makedirs(directory, exist_ok=True)

tempfile.tempdir = custom_temp_dir
os.environ["TMPDIR"] = custom_temp_dir
os.environ["TEMP"] = custom_temp_dir
os.environ["TMP"] = custom_temp_dir

# IMPORTANT: do NOT hardcode COMET_API_KEY in code.
# Export it in your shell/job environment instead.
# export COMET_API_KEY="..."
os.environ.setdefault("COMET_PROJECT", "2d3d-1disc")
os.environ.setdefault("COMET_WORKSPACE", "40uf411")
os.environ.setdefault("COMET_API_KEY", "lSTeuxfDnMITPnT8IFHY2fyWt")
os.environ["COMET_CACHE_DIR"] = comet_cache_dir
os.environ["COMET_OFFLINE_DIRECTORY"] = comet_offline_dir

try:
    import comet_ml
    print(f"Using Comet.ml temp dir: {custom_temp_dir}")
    print(f"Using Comet.ml cache dir: {comet_cache_dir}")
    print(f"Using Comet.ml offline dir: {comet_offline_dir}")
    if not os.environ.get("COMET_API_KEY"):
        print("WARNING: COMET_API_KEY is not set in the environment. Comet logging may fail.")
except ImportError:
    print("Comet ML not installed. Please install with: pip install comet_ml")
    raise

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
        "banderabrown_bin": 256,
    }
    return default_sizes.get(dataset)


@dataclass
class ScriptArguments:
    do_train: bool = True
    do_sample: bool = True
    dataset: str = "banderabrown_bin"
    batch_size: int = 32
    n_epochs: int = 150
    learning_rate: float = 1e-4  # lowered from 1e-3 to reduce early overfitting risk
    sigma_min: float = 0.0
    seed: int = 42
    output_dir: str = "outputs"
    horizontal_flip: bool = False
    image_size: int | None = None
    exp: str = "fm_exp_banderabrown_bin"

    # Logging controls
    grad_log_every: int = 100  # log gradient stats every N steps (0 disables)


def train(args: ScriptArguments):
    """Train the flow matching model on the given dataset."""

    experiment = comet_ml.Experiment(
        project_name=os.environ.get("COMET_PROJECT"),
        workspace=os.environ.get("COMET_WORKSPACE"),
    )
    experiment.log_parameters(vars(args))
    experiment.set_name(args.exp)

    output_dir = Path(args.output_dir) / "cfm" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"Using device: {device}")

    target_image_size = resolve_image_size(args.dataset, args.image_size)

    # Load dataset
    dataset = get_image_dataset(
        args.dataset,
        train=True,
        transform=get_train_transform(horizontal_flip=args.horizontal_flip, image_size=target_image_size),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Loaded {args.dataset} dataset with {len(dataset):,} samples")

    num_classes = len(dataset.classes)
    input_shape = dataset[0][0].size()
    print(f"{input_shape=}, {num_classes=}")

    # Decide whether to use class conditioning
    class_cond = (num_classes > 1)
    if not class_cond:
        print("Note: num_classes == 1, disabling class conditioning.")

    # Random real preview (NOT first 25 items)
    real_loader = DataLoader(dataset, batch_size=25, shuffle=True, drop_last=True)
    real_images, _ = next(iter(real_loader))

    # Save both raw and normalized real previews
    real_raw_path = output_dir / "real_images_raw.png"
    real_norm_path = output_dir / "real_images_norm.png"
    save_image(make_grid(real_images, nrow=5), real_raw_path, normalize=False)
    save_image(make_grid(real_images, nrow=5), real_norm_path, normalize=True)
    print(f"Saved real images grids: {real_raw_path} and {real_norm_path}")
    experiment.log_image(real_raw_path, name=f"real_preview_raw_{args.exp}")
    experiment.log_image(real_norm_path, name=f"real_preview_norm_{args.exp}")

    # Model
    flow = UNetModel(
        input_shape,
        num_channels=64,
        num_res_blocks=2,
        num_classes=num_classes,
        class_cond=class_cond,
    ).to(device)

    path_sampler = PathSampler(sigma_min=args.sigma_min)

    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.learning_rate)
    model_size_summary(flow)

    def save_sample_grid(epoch_num: int) -> Tensor:
        """Generate and save a 5x5 grid of samples for quick inspection."""

        class WrappedModel(ModelWrapper):
            def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:  # type: ignore[override]
                return self.model(x=x, t=t, **extras)

        flow.eval()
        num_samples = 25
        sample_steps = 101
        time_steps = torch.linspace(0, 1, sample_steps, device=device)

        if class_cond:
            # for visual consistency: cycle through classes
            repeats = (num_samples + num_classes - 1) // num_classes
            class_list = torch.arange(num_classes, device=device).repeat(repeats)[:num_samples]
        else:
            class_list = None

        with torch.no_grad():
            x_init = torch.randn((num_samples, *input_shape), dtype=torch.float32, device=device)
            solver = ODESolver(WrappedModel(flow))
            final_samples = solver.sample(
                x_init=x_init,
                step_size=0.05,
                method="midpoint",
                time_grid=time_steps,
                return_intermediates=False,
                y=class_list,
            )

        final_samples = final_samples.detach().cpu()

        # Save both raw and normalized versions
        raw_path = output_dir / f"samples_epoch_{epoch_num:04d}_raw.png"
        norm_path = output_dir / f"samples_epoch_{epoch_num:04d}_norm.png"
        save_image(final_samples, raw_path, nrow=5, normalize=False)
        save_image(final_samples, norm_path, nrow=5, normalize=True)
        print(f"Saved sample grids: {raw_path} and {norm_path}")

        experiment.log_image(raw_path, name=f"samples_epoch_{epoch_num:04d}_raw_{args.exp}")
        experiment.log_image(norm_path, name=f"samples_epoch_{epoch_num:04d}_norm_{args.exp}")

        flow.train()
        return final_samples

    def log_comet_gradients(experiment, model: torch.nn.Module, step: int):
        # Lightweight gradient logging; called only every args.grad_log_every steps.
        import math
        total_sq = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            gn = g.norm(2).item()
            total_sq += gn * gn
        global_l2 = math.sqrt(total_sq)
        experiment.log_metric("flow_grad_l2_step", global_l2, step=step)

    global_step = 0

    for epoch in range(args.n_epochs):
        flow.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:3d}/{args.n_epochs}")

        for x_1, y in pbar:
            x_1 = x_1.to(device)
            y = y.to(device)

            # Probability path samples
            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.size(0), device=device, dtype=x_1.dtype)
            x_t, dx_t = path_sampler.sample(x_0, x_1, t)

            optimizer.zero_grad(set_to_none=True)

            # bfloat16 autocast; no GradScaler needed
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                vf_t = flow(t=t, x=x_t, y=(y if class_cond else None))
                loss = F.mse_loss(vf_t, dx_t)

            experiment.log_metric("train_loss", float(loss.item()), step=global_step)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
            optimizer.step()

            if args.grad_log_every and (global_step % args.grad_log_every == 0):
                log_comet_gradients(experiment, flow, step=global_step)

            pbar.set_postfix({"loss": float(loss.item())})
            global_step += 1

        if (epoch + 1) % 50 == 0:
            final_samples = save_sample_grid(epoch + 1)
            real_batch, _ = next(iter(DataLoader(dataset, batch_size=final_samples.size(0), shuffle=True, drop_last=True)))
            metrics = run_eval(
                generated=final_samples,      # torch Tensor OK
                real=real_batch,              # torch Tensor OK
                cfg=EvalConfig(seed=epoch+1),
                save_json_path=str(output_dir / f"eval_epoch_{epoch+1:04d}.json"),
            )
            experiment.log_metric("eval/patch_swd", metrics.get("divergence", {}).get("patch_swd", float("nan")), step=epoch+1)
            experiment.log_metric("eval/porosity_gen_mean", metrics["generated"]["porosity"]["mean"], step=epoch+1)
            experiment.log_metric("eval/porosity_real_mean", metrics["real"]["porosity"]["mean"], step=epoch+1)
            experiment.log_image(output_dir / f"epoch_{epoch+1:04d}_tpcf.png", name=f"tpcf_epoch_{epoch+1:04d}_{args.exp}")
            experiment.log_image(output_dir / f"epoch_{epoch+1:04d}_psd.png", name=f"psd_epoch_{epoch+1:04d}_{args.exp}")
            experiment.log_image(output_dir / f"epoch_{epoch+1:04d}_pore_size.png", name=f"pore_size_epoch_{epoch+1:04d}_{args.exp}")
            experiment.log_image(output_dir / f"epoch_{epoch+1:04d}_porosity.png", name=f"porosity_epoch_{epoch+1:04d}_{args.exp}")
            experiment.log_image(output_dir / f"epoch_{epoch+1:04d}_cld.png", name=f"cld_epoch_{epoch+1:04d}_{args.exp}")
            ckpt_path = output_dir / f"ckpt_epoch_{epoch+1:04d}.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": flow.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "class_cond": class_cond,
                    "num_classes": num_classes,
                    "input_shape": tuple(input_shape),
                },
                ckpt_path,
            )
            print(f"Checkpoint saved to {ckpt_path}")

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

    dataset = get_image_dataset(
        args.dataset,
        train=False,
        transform=get_test_transform(image_size=target_image_size),
    )
    input_shape = dataset[0][0].size()
    num_classes = len(dataset.classes)
    class_cond = (num_classes > 1)

    flow = UNetModel(
        input_shape,
        num_channels=64,
        num_res_blocks=2,
        num_classes=num_classes,
        class_cond=class_cond,
    ).to(device)

    # Robust load without relying on weights_only=True
    state_dict = torch.load(output_dir / "ckpt.pth", map_location=device)
    flow.load_state_dict(state_dict)
    flow.eval()

    class WrappedModel(ModelWrapper):
        def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
            return self.model(x=x, t=t, **extras)

    samples_per_class = 10
    sample_steps = 101
    time_steps = torch.linspace(0, 1, sample_steps, device=device)

    if class_cond:
        class_list = torch.arange(num_classes, device=device).repeat(samples_per_class)
    else:
        class_list = None
        # if not class-conditional, just sample this many total
        total = samples_per_class * max(1, num_classes)

    wrapped_model = WrappedModel(flow)
    solver = ODESolver(wrapped_model)
    step_size = 0.05

    if class_cond:
        x_init = torch.randn((class_list.size(0), *input_shape), dtype=torch.float32, device=device)
    else:
        x_init = torch.randn((total, *input_shape), dtype=torch.float32, device=device)

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

    # Save raw + norm
    save_image(final_samples, output_dir / "final_samples_raw.png", nrow=(num_classes if class_cond else 10), normalize=False)
    save_image(final_samples, output_dir / "final_samples_norm.png", nrow=(num_classes if class_cond else 10), normalize=True)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    grid = make_grid(final_samples, nrow=(num_classes if class_cond else 10), normalize=True)
    ax[0].imshow(grid.permute(1, 2, 0))
    ax[0].set_title("Final samples (t = 1.0)", fontsize=16)
    ax[0].axis("off")

    def update(frame: int):
        grid = make_grid(sol[frame], nrow=(num_classes if class_cond else 10), normalize=True)
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
