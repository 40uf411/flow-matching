# Adversarial flow matching on pseudo-3D image datasets (e.g. banderabrown_bin_p3d).
import os, tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast, GradScaler
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm as std_tqdm
from transformers import HfArgumentParser

from flow_matching.datasets.image_datasets_p3D import (
    get_image_dataset,
    get_test_transform,
    get_train_transform,
)
from flow_matching.models import UNetModel
from flow_matching.solver import ModelWrapper, ODESolver
from flow_matching.utils import model_size_summary, set_seed
from flow_matching.discriminator import (
    ConditionalDiscriminator2D,
    discriminator_loss_wgan,
    generator_loss_wgan,
    gradient_penalty,
    extract_oriented_slices,
)
try:
    from flow_matching.eval_3D import eval as run_eval, EvalConfig
except Exception:
    run_eval, EvalConfig = None, None

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
        "banderabrown_bin_p3d": 64,
        "banderabrown_p3d": 64,
    }
    return default_sizes.get(dataset)


@dataclass
class ScriptArguments:
    do_train: bool = True
    do_sample: bool = True
    dataset: str = "banderabrown_bin_p3d"
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

    # -------- Pseudo-3D dataset knobs --------
    volume_size: int = 64
    k_slices: int = 30
    min_gap: int = 2
    choose_orientation: str = "random"  # "random" or "cycle"

    # -------- Policy selection --------
    train_policy: str = "fm_random_t"  # ["fm_random_t", "fm_unrolled"] Flow matching with random t vs unrolled training with fixed steps
    unroll_steps: int = 32             # number of solver steps for unrolled training
    unroll_dt: float = 0.05            # Euler dt (kept separate from inference solver)
    loss_frac: float = 0.25            # fraction of timesteps to compute loss on (unrolled)
    detach_every: int = 0              # 0 disables; else detach z every N steps to save memory


    # NEW: resume + debug cadence
    resume: bool = False                   # <-- ADD
    eval_every: int = 10                   # <-- ADD (set 1 while debugging)
    save_every: int = 10                   # <-- ADD (set 1 while debugging)

    grad_log_every: int = 0 

    # -------- Adversarial training --------
    disc_learning_rate: float = 2e-4
    lambda_gp: float = 10.0
    n_critic: int = 5

    # Discriminator architecture
    disc_base_channels: int = 64
    disc_channel_multipliers: tuple[int, ...] = (1, 2, 4, 8)
    disc_timestep_embedding_dim: int = 128
    disc_timestep_model_dim: int = 128
    disc_plane_embedding_dim: int = 128
    disc_cond_hidden_dim: int = 256
    disc_dropout: float = 0.0

    # Slice extraction for adversarial loss
    adv_slices_per_sample: int = 4
    adv_slice_strategy: str = "stratified"   # random / uniform / stratified

    # AMP
    use_amp: bool = True

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

# ---- 2D to 3D helpers ------------------------------------------- 
def _pick_loss_indices(n_steps: int, loss_frac: float, device: torch.device) -> torch.Tensor:
    k = max(1, int(round(loss_frac * n_steps)))
    idx = torch.randperm(n_steps, device=device)[:k]
    return torch.sort(idx).values

def extract_real_observed_slices(
    Y: torch.Tensor,
    M: torch.Tensor,
    plane_ids: torch.Tensor,
    slices_per_sample: int,
    strategy: str = "stratified",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract real 2D observed slices from pseudo-volume targets.

    Args:
        Y: [B,C,D,H,W]
        M: [B,1,D,H,W]
        plane_ids: [B]
        slices_per_sample: number of slices to extract per sample
        strategy: currently only used if more observed slices than requested

    Returns:
        real_slices: [B*K,C,H,W]
        plane_ids_out: [B*K]
        slice_indices_out: [B*K]
    """
    if Y.dim() != 5 or M.dim() != 5:
        raise ValueError("Expected Y [B,C,D,H,W] and M [B,1,D,H,W]")

    B, C, D, H, W = Y.shape
    device = Y.device

    real_slices = []
    out_plane_ids = []
    out_slice_indices = []

    for i in range(B):
        pid = int(plane_ids[i].item())

        if pid == 0:      # XY => observed along z
            obs = torch.nonzero(M[i, 0].sum(dim=(1, 2)) > 0, as_tuple=False).flatten()
        elif pid == 1:    # XZ => observed along y
            obs = torch.nonzero(M[i, 0].sum(dim=(0, 2)) > 0, as_tuple=False).flatten()
        elif pid == 2:    # YZ => observed along x
            obs = torch.nonzero(M[i, 0].sum(dim=(0, 1)) > 0, as_tuple=False).flatten()
        else:
            raise ValueError(f"Unknown plane id {pid}")

        if obs.numel() == 0:
            continue

        if obs.numel() <= slices_per_sample:
            chosen = obs
        else:
            if strategy == "uniform":
                pos = torch.linspace(0, obs.numel() - 1, slices_per_sample, device=device).long()
                chosen = obs[pos]
            elif strategy == "stratified":
                stratum = obs.numel() / slices_per_sample
                base = torch.arange(slices_per_sample, device=device, dtype=torch.float32) * stratum
                offs = torch.rand(slices_per_sample, device=device) * stratum
                pos = torch.clamp((base + offs).long(), max=obs.numel() - 1)
                chosen = obs[pos]
            else:  # random
                perm = torch.randperm(obs.numel(), device=device)[:slices_per_sample]
                chosen = obs[perm]

        for idx in chosen.tolist():
            if pid == 0:
                sl = Y[i, :, idx, :, :]
            elif pid == 1:
                sl = Y[i, :, :, idx, :]
            else:
                sl = Y[i, :, :, :, idx]

            real_slices.append(sl)
            out_plane_ids.append(pid)
            out_slice_indices.append(idx)

    if len(real_slices) == 0:
        raise RuntimeError("No observed real slices could be extracted.")

    return (
        torch.stack(real_slices, dim=0),
        torch.tensor(out_plane_ids, device=device, dtype=torch.long),
        torch.tensor(out_slice_indices, device=device, dtype=torch.long),
    )

def policy_generate_random_t(
    flow: torch.nn.Module,
    Y: torch.Tensor,
    M: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One-step policy:
      x0 = noise
      x1 = pseudo target on observed voxels, noise elsewhere
      x_t = linear interpolation at random t
      fake_volume = x_t + v_theta(x_t, t)

    Returns:
      fake_volume: [B,C,D,H,W]
      t: [B]
    """
    z = torch.randn_like(Y)
    x0 = z
    x1 = M * Y + (1.0 - M) * z

    t = torch.rand(Y.size(0), device=device, dtype=Y.dtype)
    t_view = t[:, None, None, None, None]
    x_t = (1.0 - t_view) * x0 + t_view * x1

    v = flow(t=t, x=x_t, y=None)
    fake_volume = x_t + v
    return fake_volume, t


def policy_generate_unrolled(
    args,
    flow: torch.nn.Module,
    Y: torch.Tensor,
    M: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unrolled generator trajectory.
    Returns selected intermediate fake volumes and their timesteps.

    Returns:
      fake_volumes: [B*K,C,D,H,W]
      t_out: [B*K]
    """
    z = torch.randn_like(Y)
    z_cur = z

    time_grid = torch.linspace(0.0, 1.0, args.unroll_steps, device=device, dtype=Y.dtype)
    chosen_idx = _pick_loss_indices(args.unroll_steps, args.loss_frac, device)
    chosen_set = set(chosen_idx.tolist())

    kept_vols = []
    kept_t = []

    for k, t_k in enumerate(time_grid):
        if args.detach_every and (k > 0) and (k % args.detach_every == 0):
            z_cur = z_cur.detach()

        t_batch = t_k.expand(Y.size(0))
        v = flow(t=t_batch, x=z_cur, y=None)
        z_cur = z_cur + args.unroll_dt * v

        if k in chosen_set:
            kept_vols.append(z_cur)
            kept_t.append(t_batch)

    if len(kept_vols) == 0:
        raise RuntimeError("No unrolled steps selected.")

    fake_volumes = torch.cat(kept_vols, dim=0)   # [B*K,C,D,H,W]
    t_out = torch.cat(kept_t, dim=0)             # [B*K]
    return fake_volumes, t_out


def generate_fake_volumes_from_policy(
    args,
    flow: torch.nn.Module,
    Y: torch.Tensor,
    M: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      fake_volumes: [B',C,D,H,W]
      fake_t: [B']
      fake_plane_ids: [B']
    """
    plane_ids = torch.randint(
        low=0,
        high=3,
        size=(Y.size(0),),
        device=device,
        dtype=torch.long,
    )

    if args.train_policy == "fm_random_t":
        fake_volumes, fake_t = policy_generate_random_t(flow, Y, M, device)
        return fake_volumes, fake_t, plane_ids

    if args.train_policy == "fm_unrolled":
        fake_volumes, fake_t = policy_generate_unrolled(args, flow, Y, M, device)
        fake_plane_ids = plane_ids.repeat(int(fake_volumes.size(0) // Y.size(0)))
        return fake_volumes, fake_t, fake_plane_ids

    raise ValueError(f"Unknown train_policy: {args.train_policy}")
# -----------------------------------------------------------------

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
        volume_size=args.volume_size,
        k_slices=args.k_slices,
        min_gap=args.min_gap,
        choose_orientation=args.choose_orientation,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Loaded {args.dataset} dataset with {len(dataset):,} samples")

    Y0, M0, o0 = dataset[0]
    input_shape = Y0.size()  # [C,D,H,W]

    # We will use class conditioning to represent orientation:
    num_classes = 1
    class_cond = False

    if not class_cond:
        print("Note: num_classes == 1, disabling class conditioning.")

    # Random real preview (NOT first 25 items)
    real_loader = DataLoader(dataset, batch_size=25, shuffle=True, drop_last=True)
    Yb, Mb, ob = next(iter(real_loader))
    # Visualize something that is likely non-empty: MIP of observed voxels
    slices = []
    for i in range(Yb.size(0)):
        M_i = Mb[i]        # [1,D,H,W]
        Y_i = Yb[i]        # [C,D,H,W]

        # Find first observed voxel along depth
        msum = M_i.sum(dim=(0,2,3))  # [D]
        if torch.any(msum > 0):
            s = torch.nonzero(msum > 0, as_tuple=False)[0].item()
        else:
            s = Y_i.size(1) // 2

        slice_xy = Y_i[:, s, :, :]   # [C,H,W]
        slices.append(slice_xy)

    real_preview = torch.stack(slices, dim=0)

    # Save both raw and normalized real previews
    real_raw_path = output_dir / "real_images_raw.png"
    real_norm_path = output_dir / "real_images_norm.png"
    save_image(make_grid(real_preview, nrow=5), real_raw_path, normalize=False)
    save_image(make_grid(real_preview, nrow=5), real_norm_path, normalize=True)
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
        use_checkpoint=True,
        use_fp16=False,
        dropout=0.0,
        resblock_updown=False,
        use_scale_shift_norm=False,
    ).to(device)

    disc = ConditionalDiscriminator2D(
        in_channels=input_shape[0],
        base_channels=args.disc_base_channels,
        channel_multipliers=args.disc_channel_multipliers,
        timestep_embedding_dim=args.disc_timestep_embedding_dim,
        timestep_model_dim=args.disc_timestep_model_dim,
        plane_embedding_dim=args.disc_plane_embedding_dim,
        cond_hidden_dim=args.disc_cond_hidden_dim,
        num_planes=3,
        dropout=args.disc_dropout,
    ).to(device)

    optimizer_G = torch.optim.AdamW(flow.parameters(), lr=args.learning_rate, betas=(0.0, 0.99))
    optimizer_D = torch.optim.AdamW(disc.parameters(), lr=args.disc_learning_rate, betas=(0.0, 0.99))

    scaler_G = GradScaler("cuda", enabled=(args.use_amp and device.type == "cuda"))
    scaler_D = GradScaler("cuda", enabled=(args.use_amp and device.type == "cuda"))

    model_size_summary(flow)
    model_size_summary(disc)

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
                y=None,
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

        for Y, M, orient_id in pbar:
            Y = Y.to(device, non_blocking=True)                 # [B,C,D,H,W]
            M = M.to(device, non_blocking=True)                 # [B,1,D,H,W]
            orient_id = orient_id.to(device, non_blocking=True) # [B]
            # 1) Discriminator updates
            for _ in range(args.n_critic):
                optimizer_D.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=(args.use_amp and device.type == "cuda")):
                    fake_volumes, fake_t, fake_plane_ids = generate_fake_volumes_from_policy(args, flow, Y, M, device)
                    fake_slices, fake_plane_ids, fake_slice_idx = extract_oriented_slices(
                        volumes=fake_volumes,
                        plane_ids=fake_plane_ids,
                        slices_per_sample=args.adv_slices_per_sample,
                        strategy=args.adv_slice_strategy,
                    )
                    real_slices, real_plane_ids, real_slice_idx = extract_real_observed_slices(
                        Y=Y,
                        M=M,
                        plane_ids=orient_id,
                        slices_per_sample=args.adv_slices_per_sample,
                        strategy=args.adv_slice_strategy,
                    )
                    # Timestep conditioning:
                    # real slices come from actual observed planes, so use t=1
                    real_t = torch.ones(real_slices.size(0), device=device, dtype=Y.dtype)
                    # fake_t is per fake volume; repeat each t for extracted slices
                    repeat_factor = args.adv_slices_per_sample
                    fake_t_slices = fake_t.repeat_interleave(repeat_factor)
                    real_scores = disc(real_slices, timesteps=real_t, plane_ids=real_plane_ids)
                    fake_scores = disc(fake_slices.detach(), timesteps=fake_t_slices, plane_ids=fake_plane_ids)
                    d_loss = discriminator_loss_wgan(real_scores, fake_scores)
                # GP in full precision
                gp = gradient_penalty(
                    discriminator=disc,
                    real_samples=real_slices.float(),
                    fake_samples=fake_slices.detach().float(),
                    timesteps=fake_t_slices.float(),
                    plane_ids=fake_plane_ids,
                    lambda_gp=args.lambda_gp,
                )
                loss_D_total = d_loss.float() + gp.float()
                scaler_D.scale(loss_D_total).backward()
                scaler_D.unscale_(optimizer_D)
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=5.0)
                scaler_D.step(optimizer_D)
                scaler_D.update()
            # 2) Generator update
            # --------------------------------------------------
            optimizer_G.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=(args.use_amp and device.type == "cuda")):
                fake_volumes, fake_t, fake_plane_ids = generate_fake_volumes_from_policy(
                    args, flow, Y, M, device
                )

                fake_slices, fake_plane_ids, fake_slice_idx = extract_oriented_slices(
                    volumes=fake_volumes,
                    plane_ids=fake_plane_ids,
                    slices_per_sample=args.adv_slices_per_sample,
                    strategy=args.adv_slice_strategy,
                )

                fake_t_slices = fake_t.repeat_interleave(args.adv_slices_per_sample)
                fake_scores = disc(fake_slices, timesteps=fake_t_slices, plane_ids=fake_plane_ids)
                loss_G = generator_loss_wgan(fake_scores)

            scaler_G.scale(loss_G).backward()
            scaler_G.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
            scaler_G.step(optimizer_G)
            scaler_G.update()

            experiment.log_metric("train/loss_G", float(loss_G.item()), step=global_step)
            experiment.log_metric("train/loss_D", float(d_loss.item()), step=global_step)
            experiment.log_metric("train/gp", float(gp.item()), step=global_step)

            if args.grad_log_every and (global_step % args.grad_log_every == 0):
                log_comet_gradients(experiment, flow, step=global_step)

            pbar.set_postfix(
                {
                    "loss_G": float(loss_G.item()),
                    "loss_D": float(d_loss.item()),
                    "gp": float(gp.item()),
                }
            )
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
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "optimizer_D_state_dict": optimizer_D.state_dict(),
                    "disc_state_dict": disc.state_dict(),
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
            # For pseudo-3D training, we do NOT have real 3D ground-truth volumes.
            # Therefore eval_3D (which expects real 3D volumes) is not applicable yet.
            # We still save sample grids for qualitative monitoring.
            _ = save_sample_grid(epoch_num)


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
    dataset_root = Path(args.data_root) / args.dataset

    dataset = get_image_dataset(
        args.dataset,
        root=dataset_root,
        train=False,
        transform=get_test_transform(image_size=target_image_size),
        volume_size=args.volume_size,
        k_slices=args.k_slices,
        min_gap=args.min_gap,
        choose_orientation=args.choose_orientation,
    )
    Y0, M0, o0 = dataset[0]
    input_shape = Y0.size()
    num_classes = 1
    class_cond = False

    flow = UNetModel(
        input_shape,
        num_channels=64,
        num_res_blocks=2,
        num_classes=num_classes,
        class_cond=class_cond,
        dims=3,
        attention_resolutions="999999"
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
    final_slices = volumes_to_slices(final_samples)
    save_image(final_slices, output_dir / "final_samples_raw.png", nrow=(num_classes if class_cond else 10), normalize=False)
    save_image(final_slices, output_dir / "final_samples_norm.png", nrow=(num_classes if class_cond else 10), normalize=True)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    grid = make_grid(final_slices, nrow=(num_classes if class_cond else 10), normalize=True)
    ax[0].imshow(grid.permute(1, 2, 0))
    ax[0].set_title("Final samples (t = 1.0)", fontsize=16)
    ax[0].axis("off")

    def update(frame: int):
        frame_slices = volumes_to_slices(sol[frame])  # sol[frame]: (B,C,D,H,W) -> (B,C,H,W)
        grid = make_grid(frame_slices, nrow=(num_classes if class_cond else 10), normalize=True)
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
