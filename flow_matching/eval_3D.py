from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from scipy import fftpack

# -------------------------
# Config
# -------------------------
@dataclass
class EvalConfig:
    seed: int = 0
    threshold: float | None = 0.0      # for tensors normalized to [-1,1], 0.0 is a good default
    patch_size: int = 16
    max_patches: int = 2048


# -------------------------
# Helpers
# -------------------------
def _to_numpy_scalar_field(x) -> np.ndarray:
    """
    Accepts:
      torch.Tensor or np.ndarray
    Shapes:
      (B,1,D,H,W) -> (B,D,H,W)
      (B,D,H,W)   -> (B,D,H,W)
      (B,1,H,W)   -> (B,H,W)
      (B,H,W)     -> (B,H,W)
    Returns numpy float32.
    """
    # Torch -> numpy
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass

    x = np.asarray(x)
    if x.ndim == 5:
        # (B, C, D, H, W)
        if x.shape[1] != 1:
            raise ValueError(f"Expected C==1, got shape {x.shape}")
        x = x[:, 0]  # (B,D,H,W)
    elif x.ndim == 4:
        # could be (B,D,H,W) or (B,1,H,W)
        if x.shape[1] == 1:  # treat as (B,1,H,W)
            x = x[:, 0]       # (B,H,W)
    elif x.ndim == 3:
        # (B,H,W)
        pass
    else:
        raise ValueError(f"Unsupported shape for eval_3D: {x.shape}")

    return x.astype(np.float32, copy=False)


def _binarize(x: np.ndarray, thr: float | None) -> np.ndarray:
    if thr is None:
        thr = 0.0
    return (x > thr).astype(np.uint8)


def _mid_slices_2d(vols_bin: np.ndarray) -> np.ndarray:
    """
    vols_bin: (B,D,H,W) uint8 -> returns (B,H,W) mid-slice
    if already (B,H,W) returns as is.
    """
    if vols_bin.ndim == 4:
        B, D, H, W = vols_bin.shape
        return vols_bin[:, D // 2, :, :]
    if vols_bin.ndim == 3:
        return vols_bin
    raise ValueError(f"Expected (B,D,H,W) or (B,H,W), got {vols_bin.shape}")


def porosity_3d(vols_bin: np.ndarray) -> dict[str, float]:
    """
    vols_bin: (B,D,H,W) or (B,H,W) uint8 in {0,1}
    Convention: solid=1, pore=0 => porosity = 1 - mean(binary)
    """
    x = vols_bin.astype(np.float32, copy=False)
    dims = tuple(range(1, x.ndim))
    por = 1.0 - x.mean(axis=dims)
    return {"mean": float(por.mean()), "std": float(por.std()), "min": float(por.min()), "max": float(por.max())}


def patch_swd_2d(real_2d: np.ndarray, gen_2d: np.ndarray, patch_size: int, max_patches: int, seed: int) -> float:
    """
    Very lightweight proxy: compare distributions of random 2D patches via sliced Wasserstein (1D projections).
    This is a simplified version that is stable and fast.
    Inputs: (B,H,W) uint8 {0,1}
    """
    rng = np.random.default_rng(seed)

    def extract_patches(imgs: np.ndarray) -> np.ndarray:
        B, H, W = imgs.shape
        P = patch_size
        n = min(max_patches, B * 64)  # cap
        patches = []
        for _ in range(n):
            b = int(rng.integers(0, B))
            y = int(rng.integers(0, H - P + 1))
            x = int(rng.integers(0, W - P + 1))
            patches.append(imgs[b, y:y+P, x:x+P].reshape(-1))
        return np.stack(patches, axis=0).astype(np.float32)

    r = extract_patches(real_2d)
    g = extract_patches(gen_2d)

    # random projections
    n_proj = 64
    proj = rng.normal(size=(r.shape[1], n_proj)).astype(np.float32)
    proj /= (np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8)

    r_p = r @ proj
    g_p = g @ proj

    # 1D Wasserstein distance per projection
    r_p.sort(axis=0)
    g_p.sort(axis=0)
    return float(np.mean(np.abs(r_p - g_p)))


def two_point_corr_2d_mean(imgs_2d: np.ndarray) -> np.ndarray:
    """
    imgs_2d: (B,H,W) uint8 {0,1}
    Returns mean 2D autocorrelation (H,W), normalized to [0,1]
    """
    imgs = imgs_2d.astype(np.float32, copy=False)
    acc = None
    for i in range(imgs.shape[0]):
        img = imgs[i]
        f = fftpack.fftn(img)
        p = np.abs(f) ** 2
        corr = np.real(fftpack.ifftn(p))
        corr = np.fft.fftshift(corr)
        corr /= (corr.max() + 1e-8)
        acc = corr if acc is None else (acc + corr)
    acc /= imgs.shape[0]
    return acc


def _save_tpcf_plot(tpcf_real: np.ndarray, tpcf_gen: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(tpcf_real)
    ax[0].set_title("TPCF real (mid-slice)")
    ax[0].axis("off")
    ax[1].imshow(tpcf_gen)
    ax[1].set_title("TPCF gen (mid-slice)")
    ax[1].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -------------------------
# Main eval entry
# -------------------------
def eval(generated, real, cfg: EvalConfig, save_json_path: str | None = None) -> dict[str, Any]:
    gen = _to_numpy_scalar_field(generated)
    rea = _to_numpy_scalar_field(real)

    # Binarize (works for both 2D and 3D; we treat intensity comparably)
    gen_bin = _binarize(gen, cfg.threshold)
    rea_bin = _binarize(rea, cfg.threshold)

    # Porosity (3D-safe)
    por_gen = porosity_3d(gen_bin)
    por_real = porosity_3d(rea_bin)

    # Slice-based metrics
    if gen_bin.ndim == 4:  # (B,D,H,W)
        gen_2d = _mid_slices_2d(gen_bin)
        rea_2d = _mid_slices_2d(rea_bin)
    else:  # (B,H,W)
        gen_2d, rea_2d = gen_bin, rea_bin

    patch = patch_swd_2d(rea_2d, gen_2d, cfg.patch_size, cfg.max_patches, seed=cfg.seed)
    tpcf_real = two_point_corr_2d_mean(rea_2d)
    tpcf_gen = two_point_corr_2d_mean(gen_2d)

    # Save a plot next to json if requested
    if save_json_path is not None:
        save_path = Path(save_json_path)
        tpcf_path = save_path.parent / f"epoch_{cfg.seed:04d}_tpcf.png"
        _save_tpcf_plot(tpcf_real, tpcf_gen, tpcf_path)

        metrics = {
            "real": {"porosity": por_real},
            "generated": {"porosity": por_gen},
            "divergence": {"patch_swd": patch},
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            import json
            json.dump(metrics, f, indent=2)
        return metrics

    return {
        "real": {"porosity": por_real},
        "generated": {"porosity": por_gen},
        "divergence": {"patch_swd": patch},
    }
