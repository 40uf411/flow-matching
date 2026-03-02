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

from flow_matching.datasets.image_datasets_3D import (
    get_image_dataset,
    get_test_transform,
    get_train_transform,
)
from flow_matching.models import UNetModel
from flow_matching.sampler import PathSampler
from flow_matching.solver import ModelWrapper, ODESolver
from flow_matching.utils import model_size_summary, set_seed

from flow_matching.eval_3D import eval as run_eval, EvalConfig

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
os.environ.setdefault("COMET_PROJECT", "3d3d")
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


def volumes_to_slices(
    x: torch.Tensor,
    axis: int = 2,              # 2 means depth dimension for (B,C,D,H,W)
    index: int | None = None,   # None -> middle slice
) -> torch.Tensor:
    """
    Convert 3D volumes (B,C,D,H,W) into 2D slices (B,C,H,W) for visualization.
    """
    if x.dim() != 5:
        raise ValueError(f"Expected 5D tensor (B,C,D,H,W), got {tuple(x.shape)}")

    B, C, D, H, W = x.shape
    if axis != 2:
        raise NotImplementedError("This helper assumes axis=2 corresponds to depth (D).")

    d = (D // 2) if index is None else index
    if not (0 <= d < D):
        raise ValueError(f"Slice index {d} out of range for D={D}")

    return x[:, :, d, :, :]  # (B,C,H,W)

def volumes_to_mip(x: torch.Tensor) -> torch.Tensor:
    """
    Max intensity projection over depth: (B,C,D,H,W) -> (B,C,H,W)
    """
    if x.dim() != 5:
        raise ValueError(f"Expected 5D tensor, got {tuple(x.shape)}")
    return x.max(dim=2).values


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
    data_root: str = "datasets_3d"          # <-- ADD (parent folder containing dataset subfolders)
    batch_size: int = 32
    n_epochs: int = 150
    learning_rate: float = 1e-4
    sigma_min: float = 0.0
    seed: int = 42
    output_dir: str = "outputs"
    horizontal_flip: bool = False
    image_size: int | None = None
    exp: str = "fm_exp_banderabrown_bin"

    # NEW: resume + debug cadence
    resume: bool = True                   # <-- ADD
    eval_every: int = 1                    # <-- ADD (set 1 while debugging)
    save_every: int = 1                    # <-- ADD (set 1 while debugging)

    grad_log_every: int = 0   # ✅ ADD THIS

def _find_resume_checkpoint(output_dir: Path) -> Path | None:
    """
    Prefer latest ckpt_epoch_XXXX.pth if present; otherwise fall back to ckpt.pth.
    """
    epoch_ckpts = sorted(output_dir.glob("ckpt_epoch_*.pth"))
    if epoch_ckpts:
        return epoch_ckpts[-1]
    last = output_dir / "ckpt.pth"
    if last.exists():
        return last
    return None


def _load_checkpoint_maybe(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[int, int]:
    """
    Loads either:
      - full checkpoint dict with keys: model_state_dict, optimizer_state_dict, epoch, global_step
      - OR a raw state_dict (OrderedDict)
    Returns: (start_epoch, global_step)
    """
    obj = torch.load(ckpt_path, map_location=device)

    start_epoch = 0
    global_step = 0

    # Case A: full checkpoint
    if isinstance(obj, dict) and ("model_state_dict" in obj or "state_dict" in obj):
        sd = obj.get("model_state_dict", obj.get("state_dict"))
        model.load_state_dict(sd, strict=True)

        if optimizer is not None and "optimizer_state_dict" in obj:
            try:
                optimizer.load_state_dict(obj["optimizer_state_dict"])
            except Exception as e:
                print(f"WARNING: could not load optimizer state ({e}). Continuing with fresh optimizer.")

        start_epoch = int(obj.get("epoch", 0))
        global_step = int(obj.get("global_step", 0))
        return start_epoch, global_step

    # Case B: raw state_dict
    model.load_state_dict(obj, strict=True)
    return 0, 0


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
    dataset_root = Path(args.data_root) / args.dataset
    dataset = get_image_dataset(
        args.dataset,
        root=dataset_root,
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
    real_vols, _ = next(iter(real_loader))          # (B,C,D,H,W)
    real_slices = volumes_to_slices(real_vols)      # (B,C,H,W)

    # Save both raw and normalized real previews
    real_raw_path = output_dir / "real_images_raw.png"
    real_norm_path = output_dir / "real_images_norm.png"
    save_image(make_grid(real_slices, nrow=5), real_raw_path, normalize=False)
    save_image(make_grid(real_slices, nrow=5), real_norm_path, normalize=True)
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
        dims=3,
        attention_resolutions="999999",
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
        final_slices = volumes_to_slices(final_samples)
        save_image(final_slices, raw_path, nrow=5, normalize=False)
        save_image(final_slices, norm_path, nrow=5, normalize=True)
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

    start_epoch = 0
    global_step = 0

    if args.resume:
        resume_path = _find_resume_checkpoint(output_dir)
        if resume_path is None:
            print("Resume requested, but no checkpoint found. Starting from scratch.")
        else:
            start_epoch, global_step = _load_checkpoint_maybe(
                ckpt_path=resume_path,
                model=flow,
                optimizer=optimizer,
                device=device,
            )
            # IMPORTANT: start_epoch returned is the last completed epoch number,
            # so we continue from there (epoch index start_epoch)
            print(f"Resumed from {resume_path} | start_epoch={start_epoch}, global_step={global_step}")


    for epoch in range(start_epoch, args.n_epochs):
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

        # --- end of each epoch ---
        epoch_num = epoch + 1

        # 1) Save checkpoint (frequent during debugging)
        if args.save_every > 0 and (epoch_num % args.save_every == 0):
            ckpt_path = output_dir / f"ckpt_epoch_{epoch_num:04d}.pth"
            torch.save(
                {
                    "epoch": epoch_num,
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
            # Also keep a rolling pointer
            torch.save(flow.state_dict(), output_dir / "ckpt.pth")
            print(f"Checkpoint saved to {ckpt_path} (+ updated ckpt.pth)")

        # 2) Eval cadence (frequent during debugging)
        if args.eval_every > 0 and (epoch_num % args.eval_every == 0):
            final_samples = save_sample_grid(epoch_num)

            real_batch, _ = next(
                iter(DataLoader(dataset, batch_size=final_samples.size(0), shuffle=True, drop_last=True))
            )

            metrics = run_eval(
                generated=final_samples,  # torch Tensor
                real=real_batch,          # torch Tensor
                cfg=EvalConfig(seed=epoch_num),
                save_json_path=str(output_dir / f"eval_epoch_{epoch_num:04d}.json"),
            )

            # log metrics/images (keep your existing logging lines)
            experiment.log_metric("eval/patch_swd", metrics.get("divergence", {}).get("patch_swd", float("nan")), step=epoch_num)
            experiment.log_metric("eval/porosity_gen_mean", metrics["generated"]["porosity"]["mean"], step=epoch_num)
            experiment.log_metric("eval/porosity_real_mean", metrics["real"]["porosity"]["mean"], step=epoch_num)

            # log plots if they exist
            for name in ["tpcf", "psd", "pore_size", "porosity", "cld"]:
                p = output_dir / f"epoch_{epoch_num:04d}_{name}.png"
                if p.exists():
                    experiment.log_image(p, name=f"{name}_epoch_{epoch_num:04d}_{args.exp}")



    torch.save(flow.state_dict(), output_dir / "ckpt.pth")
    print(f"Final checkpoint saved to {output_dir / 'ckpt.pth'}")



def generate_samples_and_save_animation(args: ScriptArguments):
    output_dir = Path(args.output_dir) / "cfm" / args.dataset
    assert output_dir.is_dir(), f"Output directory {output_dir} does not exist"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"Using device: {device}")

    # -------------------------
    # HARD-CODED MODEL SETTINGS
    # -------------------------
    input_shape = (1, 64, 64, 64)
    num_classes = 1
    class_cond = False
    class_list = None

    flow = UNetModel(
        input_shape,
        num_channels=64,
        num_res_blocks=2,
        num_classes=num_classes,
        class_cond=class_cond,
        dims=3,
        attention_resolutions="999999",
    ).to(device)

    state_dict = torch.load(output_dir / "ckpt.pth", map_location=device)
    flow.load_state_dict(state_dict, strict=True)
    flow.eval()

    class WrappedModel(ModelWrapper):
        def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
            return self.model(x=x, t=t, **extras)

    wrapped_model = WrappedModel(flow)
    solver = ODESolver(wrapped_model)

    # -------------------------
    # SAMPLING SETTINGS
    # -------------------------
    total_samples = 100          # how many 3D volumes you want in total
    batch_samples = 20            # chunk size to avoid OOM (try 2/4/5/8)
    sample_steps = 101
    time_steps = torch.linspace(0, 1, sample_steps, device=device)
    step_size = 0.05

    # For GIF only (keep small!)
    make_gif = True
    gif_samples = 10             # number of volumes in GIF (small)
    gif_return_intermediates = True

    # -------------------------
    # 1) Generate ALL final samples in batches (NO intermediates)
    # -------------------------
    all_final = []
    remaining = total_samples

    with torch.no_grad():
        while remaining > 0:
            b = min(batch_samples, remaining)
            x_init = torch.randn((b, *input_shape), dtype=torch.float32, device=device)

            # key: return_intermediates=False to save memory
            final = solver.sample(
                x_init=x_init,
                step_size=step_size,
                method="midpoint",
                time_grid=time_steps,
                return_intermediates=False,
                y=class_list,
            )

            all_final.append(final.detach().cpu())
            remaining -= b
            print(f"Sampled {total_samples - remaining}/{total_samples}")

    final_samples = torch.cat(all_final, dim=0)  # (N,C,D,H,W)

    # Save ALL volumes to npy
    import numpy as np
    npy_path = output_dir / "generated_samples.npy"
    np.save(npy_path, final_samples.numpy())
    print(f"Saved generated volumes to {npy_path} with shape {tuple(final_samples.shape)}")

    # Save 2D grids of middle slices
    final_slices = volumes_to_slices(final_samples)  # (N,C,H,W)
    # choose a square-ish grid: for 100 => 10x10
    nrow = int(round(total_samples ** 0.5))
    save_image(final_slices, output_dir / "final_samples_raw.png", nrow=nrow, normalize=False)
    save_image(final_slices, output_dir / "final_samples_norm.png", nrow=nrow, normalize=True)

    # -------------------------
    # 2) Optional GIF: do a SMALL run with intermediates
    # -------------------------
    if make_gif:
        with torch.no_grad():
            x_init_gif = torch.randn((gif_samples, *input_shape), dtype=torch.float32, device=device)
            sol = solver.sample(
                x_init=x_init_gif,
                step_size=step_size,
                method="midpoint",
                time_grid=time_steps,
                return_intermediates=gif_return_intermediates,  # True for GIF
                y=None,
            ).detach().cpu()  # (T,B,C,D,H,W)

        # Build GIF from slices
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        grid0 = make_grid(volumes_to_slices(sol[-1]), nrow=gif_samples, normalize=True)
        ax[0].imshow(grid0.permute(1, 2, 0))
        ax[0].set_title("Final samples (t=1.0)", fontsize=16)
        ax[0].axis("off")

        def update(frame: int):
            frame_slices = volumes_to_slices(sol[frame])
            grid = make_grid(frame_slices, nrow=gif_samples, normalize=True)
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
