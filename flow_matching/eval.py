import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.ndimage import distance_transform_edt
from scipy.stats import wasserstein_distance


# =========================
# Config
# =========================

@dataclass
class EvalConfig:
    seed: int = 0
    threshold: float = 0.5   # binarization threshold
    max_patches: int = 5000
    patch_size: int = 16


# =========================
# Utilities
# =========================

def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def binarize(imgs: np.ndarray, thr: float) -> np.ndarray:
    return (imgs > thr).astype(np.uint8)


# =========================
# Metrics
# =========================

def patch_swd(real, gen, patch_size, max_patches):
    """Sliced Wasserstein on image patches."""
    def extract(x):
        patches = []
        for img in x:
            H, W = img.shape
            for _ in range(10):
                i = np.random.randint(0, H - patch_size)
                j = np.random.randint(0, W - patch_size)
                patches.append(img[i:i+patch_size, j:j+patch_size].ravel())
                if len(patches) >= max_patches:
                    break
        return np.array(patches)

    r = extract(real)
    g = extract(gen)
    swd = np.mean([wasserstein_distance(r[:, i], g[:, i]) for i in range(r.shape[1])])
    return float(swd)


def porosity(binary):
    return 1.0 - binary.mean(axis=(1, 2))


def two_point_corr(binary, max_r=64):
    img = binary[0]
    f = fftpack.fftn(img)
    corr = fftpack.ifftn(f * np.conj(f)).real
    corr = fftpack.fftshift(corr)
    center = corr.shape[0] // 2
    return corr[center, center:center+max_r]


def power_spectrum(img):
    f = fftpack.fftn(img)
    ps = np.abs(f)**2
    return np.mean(ps, axis=0)


def chord_length(binary):
    lengths = []
    for img in binary:
        for row in img:
            runs = np.diff(np.where(np.concatenate(([row[0]],
                                                     row[:-1] != row[1:],
                                                     [True])))[0])[::2]
            lengths.extend(runs)
    return np.array(lengths)


def pore_size(binary):
    sizes = []
    for img in binary:
        dt = distance_transform_edt(img == 0)
        sizes.extend(dt[dt > 0])
    return np.array(sizes)


def minkowski(binary):
    area = binary.mean()
    perimeter = np.sum(np.abs(np.diff(binary, axis=1))) + np.sum(np.abs(np.diff(binary, axis=2)))
    euler = area - perimeter
    return area, perimeter, euler


# =========================
# Plot helpers
# =========================

def save_hist(real, gen, title, path):
    plt.figure()
    plt.hist(real, bins=50, alpha=0.6, label="Real", density=True)
    plt.hist(gen, bins=50, alpha=0.6, label="Generated", density=True)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_curve(real, gen, title, ylabel, path):
    plt.figure()
    plt.plot(real, label="Real")
    plt.plot(gen, label="Generated")
    plt.legend()
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# =========================
# Entry point
# =========================

def eval(
    generated: torch.Tensor,
    real: torch.Tensor,
    cfg: EvalConfig,
    save_json_path: str,
) -> Dict[str, Any]:

    np.random.seed(cfg.seed)

    save_json_path = Path(save_json_path)
    out_dir = save_json_path.parent 
    ensure_dir(out_dir)

    gen = to_numpy(generated.squeeze(1))
    real = to_numpy(real.squeeze(1))

    gen_bin = binarize(gen, cfg.threshold)
    real_bin = binarize(real, cfg.threshold)

    # ---- Metrics
    metrics = {
        "divergence": {
            "patch_swd": patch_swd(real, gen, cfg.patch_size, cfg.max_patches),
        },
        "generated": {},
        "real": {},
    }

    # Porosity
    por_gen = porosity(gen_bin)
    por_real = porosity(real_bin)
    metrics["generated"]["porosity"] = {
        "mean": float(por_gen.mean()),
        "std": float(por_gen.std()),
    }
    metrics["real"]["porosity"] = {
        "mean": float(por_real.mean()),
        "std": float(por_real.std()),
    }
    save_hist(por_real, por_gen, "Porosity", out_dir / f"epoch_{cfg.seed:04d}_porosity.png")

    # Two-point correlation
    tpcf_real = two_point_corr(real_bin)
    tpcf_gen = two_point_corr(gen_bin)
    save_curve(tpcf_real, tpcf_gen, "Two-Point Correlation", "C(r)",
               out_dir / f"epoch_{cfg.seed:04d}_tpcf.png")

    # Power spectrum
    ps_real = power_spectrum(real[0])
    ps_gen = power_spectrum(gen[0])
    save_curve(ps_real, ps_gen, "Power Spectrum", "PSD",
               out_dir / f"epoch_{cfg.seed:04d}_psd.png")

    # Chord length
    cl_real = chord_length(real_bin)
    cl_gen = chord_length(gen_bin)
    save_hist(cl_real, cl_gen, "Chord Length Distribution",
              out_dir / f"epoch_{cfg.seed:04d}_cld.png")

    # Pore size
    ps_real = pore_size(real_bin)
    ps_gen = pore_size(gen_bin)
    save_hist(ps_real, ps_gen, "Pore Size Distribution",
              out_dir / f"epoch_{cfg.seed:04d}_pore_size.png")

    # Minkowski
    m_gen = minkowski(gen_bin)
    m_real = minkowski(real_bin)
    metrics["generated"]["minkowski"] = {
        "area": float(m_gen[0]),
        "perimeter": float(m_gen[1]),
        "euler": float(m_gen[2]),
    }
    metrics["real"]["minkowski"] = {
        "area": float(m_real[0]),
        "perimeter": float(m_real[1]),
        "euler": float(m_real[2]),
    }

    # Save JSON
    with open(save_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
