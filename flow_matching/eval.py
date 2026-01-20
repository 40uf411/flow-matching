# eval.py
# Self-contained evaluation utilities for 2D porous / microstructure images.
# Provides eval(...) entry point that runs:
# - Patch distribution divergence (Sliced Wasserstein on patches)
# - Pore statistics (porosity, components, etc.)
# - Two-point correlation S2(r) via FFT
# - Lineal path L(r)
# - Chord length distribution
# - Pore size distribution (distance-transform based)
# - Minkowski functionals (area, perimeter, Euler) in 2D
# - Power spectrum (radially averaged)

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

# Optional SciPy/skimage, with clear errors if missing for specific functions.
try:
    from scipy import ndimage as ndi
except Exception as e:
    ndi = None

try:
    from skimage.filters import threshold_otsu
    from skimage.measure import perimeter as sk_perimeter
except Exception:
    threshold_otsu = None
    sk_perimeter = None


ArrayLike = Union[np.ndarray, "torch.Tensor"]  # torch is optional at runtime


# ----------------------------
# Helpers: input handling
# ----------------------------
def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert torch Tensor or numpy array to numpy float32 array on CPU."""
    if "torch" in str(type(x)):
        import torch  # local import

        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _ensure_nhw(x: np.ndarray) -> np.ndarray:
    """
    Convert input to shape (N, H, W) float32.
    Accepts (H,W), (C,H,W), (N,H,W), (N,C,H,W).
    If C>1, converts to grayscale by mean over channel.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return x[None, ...]
    if x.ndim == 3:
        # could be (C,H,W) or (N,H,W)
        if x.shape[0] in (1, 3) and x.shape[1] >= 16 and x.shape[2] >= 16:
            # assume (C,H,W)
            return x.mean(axis=0, keepdims=True)
        else:
            # assume (N,H,W)
            return x
    if x.ndim == 4:
        # (N,C,H,W)
        return x.mean(axis=1)
    raise ValueError(f"Unsupported input shape {x.shape}. Expected 2D/3D/4D.")


def _normalize_to_unit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize each image to [0,1] using min-max per image.
    This is for evaluation stability when model outputs are unconstrained.
    """
    x = x.astype(np.float32)
    mn = x.reshape(x.shape[0], -1).min(axis=1)[:, None, None]
    mx = x.reshape(x.shape[0], -1).max(axis=1)[:, None, None]
    return (x - mn) / (mx - mn + eps)


def binarize(
    x: np.ndarray,
    threshold: str = "otsu",
    fixed_threshold: Optional[float] = None,
    invert: bool = False,
) -> np.ndarray:
    """
    Convert grayscale images (N,H,W) in [0,1] to boolean pore mask.
    Default threshold is Otsu per-image (requires skimage).
    If invert=True, flips pore/solid.
    """
    x = np.asarray(x, dtype=np.float32)
    if fixed_threshold is not None:
        ths = np.full((x.shape[0],), float(fixed_threshold), dtype=np.float32)
    else:
        if threshold == "otsu":
            if threshold_otsu is None:
                raise RuntimeError("skimage is required for Otsu thresholding (skimage.filters.threshold_otsu).")
            ths = np.array([threshold_otsu(xi) for xi in x], dtype=np.float32)
        elif threshold == "mean":
            ths = x.reshape(x.shape[0], -1).mean(axis=1).astype(np.float32)
        else:
            raise ValueError(f"Unknown threshold method: {threshold}")

    pore = x > ths[:, None, None]
    if invert:
        pore = ~pore
    return pore


# ----------------------------
# Patch distribution divergence
# ----------------------------
def _extract_patches(
    imgs: np.ndarray,
    patch: int,
    n_patches: int,
    seed: int,
) -> np.ndarray:
    """
    Randomly sample n_patches patches across the batch.
    Returns array (n_patches, patch*patch) float32.
    """
    rng = np.random.default_rng(seed)
    N, H, W = imgs.shape
    if H < patch or W < patch:
        raise ValueError(f"Patch size {patch} larger than image {H}x{W}")

    patches = np.empty((n_patches, patch * patch), dtype=np.float32)
    for i in range(n_patches):
        n = int(rng.integers(0, N))
        y = int(rng.integers(0, H - patch + 1))
        x = int(rng.integers(0, W - patch + 1))
        p = imgs[n, y : y + patch, x : x + patch].astype(np.float32)
        patches[i] = p.reshape(-1)
    return patches


def sliced_wasserstein_distance(
    A: np.ndarray,
    B: np.ndarray,
    n_projections: int = 128,
    seed: int = 0,
) -> float:
    """
    Sliced Wasserstein distance between two point clouds A and B (n,d).
    Uses random 1D projections and averages 1D Wasserstein (L2 between sorted projections).
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays (n,d).")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Dim mismatch: {A.shape} vs {B.shape}")

    d = A.shape[1]
    # Random unit vectors
    dirs = rng.normal(size=(n_projections, d)).astype(np.float32)
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)

    sw = 0.0
    for v in dirs:
        a = A @ v
        b = B @ v
        a.sort()
        b.sort()
        m = min(a.size, b.size)
        sw += float(np.mean((a[:m] - b[:m]) ** 2))
    return float(math.sqrt(sw / n_projections))


def patch_distribution_divergence(
    gen_imgs: np.ndarray,
    real_imgs: np.ndarray,
    patch: int = 32,
    n_patches: int = 4096,
    n_projections: int = 128,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Patch distribution divergence using Sliced Wasserstein Distance on raw patch vectors.
    Returns dict with SWD.
    """
    g = _extract_patches(gen_imgs, patch, n_patches, seed=seed + 1)
    r = _extract_patches(real_imgs, patch, n_patches, seed=seed + 2)

    # Optional: standardize patches to reduce sensitivity to global brightness
    g = (g - g.mean(axis=1, keepdims=True)) / (g.std(axis=1, keepdims=True) + 1e-8)
    r = (r - r.mean(axis=1, keepdims=True)) / (r.std(axis=1, keepdims=True) + 1e-8)

    swd = sliced_wasserstein_distance(g, r, n_projections=n_projections, seed=seed + 3)
    return {"patch_swd": float(swd)}


# ----------------------------
# Pore statistics and descriptors
# ----------------------------
def porosity(pore: np.ndarray) -> np.ndarray:
    """Porosity per image."""
    return pore.reshape(pore.shape[0], -1).mean(axis=1).astype(np.float32)


def connected_components_stats(pore: np.ndarray, connectivity: int = 1) -> Dict[str, np.ndarray]:
    """
    Basic connected component stats on pore phase:
    - num_components
    - mean_component_area (pixels)
    Requires scipy.ndimage.
    """
    if ndi is None:
        raise RuntimeError("scipy is required for connected component stats (scipy.ndimage).")

    struct = ndi.generate_binary_structure(2, connectivity)
    nums = np.zeros((pore.shape[0],), dtype=np.int32)
    means = np.zeros((pore.shape[0],), dtype=np.float32)
    for i in range(pore.shape[0]):
        lab, n = ndi.label(pore[i], structure=struct)
        nums[i] = n
        if n > 0:
            counts = np.bincount(lab.ravel())[1:]  # skip background
            means[i] = float(np.mean(counts))
        else:
            means[i] = 0.0
    return {"num_components": nums, "mean_component_area": means}


def specific_surface_2d(pore: np.ndarray) -> np.ndarray:
    """
    2D "specific surface" proxy: perimeter(pore)/area_total per image.
    Uses skimage.measure.perimeter if available, else a simple finite-diff perimeter estimate.
    """
    N, H, W = pore.shape
    out = np.zeros((N,), dtype=np.float32)

    if sk_perimeter is not None:
        for i in range(N):
            per = float(sk_perimeter(pore[i], neighborhood=8))
            out[i] = per / float(H * W)
        return out

    # Fallback: count edge transitions (4-neighborhood)
    for i in range(N):
        m = pore[i].astype(np.uint8)
        # transitions horizontally and vertically
        tx = np.abs(np.diff(m, axis=1)).sum()
        ty = np.abs(np.diff(m, axis=0)).sum()
        out[i] = float(tx + ty) / float(H * W)
    return out


def euler_characteristic_2d(pore: np.ndarray) -> np.ndarray:
    """
    Euler characteristic (approx) using connectivity-based formula:
    chi = (#components in pore) - (#components in solid) + 1
    This is a practical proxy for 2D binary images.
    Requires scipy.ndimage.
    """
    if ndi is None:
        raise RuntimeError("scipy is required for Euler characteristic proxy (scipy.ndimage).")

    chi = np.zeros((pore.shape[0],), dtype=np.float32)
    struct = ndi.generate_binary_structure(2, 1)
    for i in range(pore.shape[0]):
        _, n_p = ndi.label(pore[i], structure=struct)
        _, n_s = ndi.label(~pore[i], structure=struct)
        # Background component included in solid; adjust with +1
        chi[i] = float(n_p - (n_s - 1))
    return chi


def pore_size_distribution(pore: np.ndarray, bins: int = 50, max_radius: Optional[float] = None) -> Dict[str, Any]:
    """
    Pore size distribution using Euclidean distance transform (EDT).
    Returns histogram of radii (pixels) over pore pixels.
    Requires scipy.ndimage.
    """
    if ndi is None:
        raise RuntimeError("scipy is required for distance transform (scipy.ndimage).")

    # Collect radii across batch
    radii = []
    for i in range(pore.shape[0]):
        dt = ndi.distance_transform_edt(pore[i])
        vals = dt[pore[i]]
        if vals.size > 0:
            radii.append(vals.astype(np.float32))
    if not radii:
        return {"bins": None, "hist": None, "count": 0}

    radii = np.concatenate(radii, axis=0)
    if max_radius is None:
        max_radius = float(np.percentile(radii, 99.5))  # robust cap

    hist, edges = np.histogram(radii, bins=bins, range=(0.0, max_radius), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "bins": centers.astype(np.float32),
        "hist": hist.astype(np.float32),
        "count": int(radii.size),
        "max_radius": float(max_radius),
    }


def chord_length_distribution(
    pore: np.ndarray,
    direction: str = "both",
    max_len: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Chord length distribution (run-lengths) in pore phase along x/y.
    direction: "x", "y", or "both"
    Returns histogram over chord lengths (pixels), pooled across batch.
    """
    N, H, W = pore.shape
    if max_len is None:
        max_len = max(H, W)

    lengths = []

    def runs_1d(arr: np.ndarray) -> np.ndarray:
        # arr is boolean 1D
        if arr.size == 0:
            return np.array([], dtype=np.int32)
        # Find run boundaries
        d = np.diff(arr.astype(np.int8))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
        if arr[0]:
            starts = np.r_[0, starts]
        if arr[-1]:
            ends = np.r_[ends, arr.size]
        return (ends - starts).astype(np.int32)

    for i in range(N):
        img = pore[i]
        if direction in ("x", "both"):
            for r in range(H):
                lengths.append(runs_1d(img[r, :]))
        if direction in ("y", "both"):
            for c in range(W):
                lengths.append(runs_1d(img[:, c]))

    lengths = np.concatenate([x for x in lengths if x.size > 0], axis=0) if lengths else np.array([], dtype=np.int32)
    if lengths.size == 0:
        return {"lengths": None, "pmf": None, "count": 0}

    # PMF over lengths 1..max_len
    counts = np.bincount(lengths, minlength=max_len + 1).astype(np.float64)
    counts[0] = 0.0
    pmf = counts / (counts.sum() + 1e-12)
    xs = np.arange(max_len + 1, dtype=np.int32)
    return {"lengths": xs.astype(np.int32), "pmf": pmf.astype(np.float32), "count": int(lengths.size)}


def lineal_path_function(
    pore: np.ndarray,
    max_r: int = 128,
    direction: str = "both",
) -> Dict[str, Any]:
    """
    Lineal path L(r): probability that a random line segment of length r lies entirely in pore phase.
    Computed along x/y by scanning 1D runs and accumulating contributions.
    Returns L(r) for r=1..max_r.
    """
    N, H, W = pore.shape
    max_r = int(min(max_r, max(H, W)))

    def accumulate_from_runs(run_lengths: np.ndarray, L: np.ndarray):
        # For a run of length L0, number of segments of length r fully inside is max(L0 - r + 1, 0)
        # We accumulate over r.
        for L0 in run_lengths:
            m = min(L0, max_r)
            # For r=1..m: add (L0 - r + 1)
            # Vectorized:
            r = np.arange(1, m + 1, dtype=np.int32)
            L[1 : m + 1] += (L0 - r + 1)

    # Count total possible segments of length r (denominator)
    # For each row length W: number of segments of length r is (W - r + 1) if positive.
    # Similarly for columns length H.
    L_num = np.zeros((max_r + 1,), dtype=np.float64)
    L_den = np.zeros((max_r + 1,), dtype=np.float64)

    for i in range(N):
        img = pore[i]
        if direction in ("x", "both"):
            for r0 in range(H):
                row = img[r0, :]
                # run lengths in row
                # compute runs
                d = np.diff(row.astype(np.int8))
                starts = np.where(d == 1)[0] + 1
                ends = np.where(d == -1)[0] + 1
                if row[0]:
                    starts = np.r_[0, starts]
                if row[-1]:
                    ends = np.r_[ends, row.size]
                runs = (ends - starts).astype(np.int32)
                if runs.size:
                    accumulate_from_runs(runs, L_num)

            for rr in range(1, max_r + 1):
                L_den[rr] += H * max(W - rr + 1, 0)

        if direction in ("y", "both"):
            for c0 in range(W):
                col = img[:, c0]
                d = np.diff(col.astype(np.int8))
                starts = np.where(d == 1)[0] + 1
                ends = np.where(d == -1)[0] + 1
                if col[0]:
                    starts = np.r_[0, starts]
                if col[-1]:
                    ends = np.r_[ends, col.size]
                runs = (ends - starts).astype(np.int32)
                if runs.size:
                    accumulate_from_runs(runs, L_num)

            for rr in range(1, max_r + 1):
                L_den[rr] += W * max(H - rr + 1, 0)

    L = np.zeros((max_r + 1,), dtype=np.float32)
    valid = L_den > 0
    L[valid] = (L_num[valid] / (L_den[valid] + 1e-12)).astype(np.float32)
    r = np.arange(max_r + 1, dtype=np.int32)
    return {"r": r.astype(np.int32), "L": L.astype(np.float32)}


# ----------------------------
# Two-point correlation S2(r) via FFT
# ----------------------------
def _radial_average_2d(img: np.ndarray, nbins: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radial average of a 2D image around its center.
    Returns (r_centers, mean_values).
    """
    H, W = img.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.indices((H, W))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)

    rmax = rr.max()
    bins = np.linspace(0.0, rmax, nbins + 1, dtype=np.float32)
    inds = np.digitize(rr.ravel(), bins) - 1
    vals = img.ravel()

    sums = np.bincount(inds, weights=vals, minlength=nbins).astype(np.float64)
    cnts = np.bincount(inds, minlength=nbins).astype(np.float64)
    means = (sums / (cnts + 1e-12)).astype(np.float32)

    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, means


def two_point_correlation(pore: np.ndarray, nbins: int = 128) -> Dict[str, Any]:
    """
    Two-point correlation S2(r) of pore phase using FFT-based autocorrelation.
    For each image:
      S2 = <I(x) I(x+r)> where I is indicator of pore.
    We average S2 over batch and radial-average it.
    """
    pore_f = pore.astype(np.float32)
    N, H, W = pore_f.shape

    acc = np.zeros((H, W), dtype=np.float64)
    for i in range(N):
        I = pore_f[i]
        F = np.fft.fft2(I)
        # autocorrelation via inverse FFT of |F|^2
        ac = np.fft.ifft2(F * np.conj(F)).real
        ac = np.fft.fftshift(ac)
        ac /= float(H * W)  # normalize
        acc += ac

    acc = (acc / max(N, 1)).astype(np.float32)
    r, s2 = _radial_average_2d(acc, nbins=nbins)
    return {"r": r.astype(np.float32), "S2": s2.astype(np.float32)}


# ----------------------------
# Power spectrum (radially averaged)
# ----------------------------
def power_spectrum(imgs: np.ndarray, nbins: int = 128) -> Dict[str, Any]:
    """
    Radially averaged power spectrum of grayscale images.
    Uses FFT magnitude squared, averaged over batch.
    """
    imgs = imgs.astype(np.float32)
    N, H, W = imgs.shape
    acc = np.zeros((H, W), dtype=np.float64)

    for i in range(N):
        x = imgs[i] - float(imgs[i].mean())
        F = np.fft.fft2(x)
        P = (F * np.conj(F)).real
        P = np.fft.fftshift(P)
        acc += P

    acc = (acc / max(N, 1)).astype(np.float32)
    r, ps = _radial_average_2d(acc, nbins=nbins)
    return {"k": r.astype(np.float32), "P": ps.astype(np.float32)}


# ----------------------------
# Minkowski functionals (2D)
# ----------------------------
def minkowski_functionals_2d(pore: np.ndarray) -> Dict[str, np.ndarray]:
    """
    2D Minkowski functionals (densities / proxies):
    - area fraction (porosity)
    - perimeter density (perimeter/area_total)
    - Euler characteristic density (chi/area_total) via proxy
    """
    N, H, W = pore.shape
    area_frac = porosity(pore)
    per_dens = specific_surface_2d(pore)
    chi = euler_characteristic_2d(pore)
    chi_dens = (chi / float(H * W)).astype(np.float32)
    return {
        "area_fraction": area_frac.astype(np.float32),
        "perimeter_density": per_dens.astype(np.float32),
        "euler_density": chi_dens,
    }


# ----------------------------
# Aggregation helpers
# ----------------------------
def _mean_std(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x)
    return {"mean": float(np.mean(x)), "std": float(np.std(x))}


def _maybe_reduce_vectors(d: Dict[str, Any], max_len: int = 256) -> Dict[str, Any]:
    """
    Keep vectors as-is, but optionally truncate overly long vectors for logging/JSON.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > max_len:
            out[k] = v[:max_len]
            out[k + "_truncated"] = True
        else:
            out[k] = v
    return out


# ----------------------------
# Public entry point
# ----------------------------
@dataclass
class EvalConfig:
    # Binarization
    threshold: str = "otsu"
    fixed_threshold: Optional[float] = None
    invert: bool = False

    # Patch divergence
    patch: int = 32
    n_patches: int = 4096
    swd_projections: int = 128

    # Correlations / spectra
    radial_bins: int = 128
    lineal_max_r: int = 128
    chord_max_len: Optional[int] = None
    psd_bins: int = 128

    # Pore size distribution
    psd_bins_count: int = 50
    psd_max_radius: Optional[float] = None

    # Randomness
    seed: int = 0


def eval(
    generated: ArrayLike,
    real: Optional[ArrayLike] = None,
    cfg: Optional[EvalConfig] = None,
    save_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Access point to run all evaluations at once.

    Args:
        generated: generated samples (torch or numpy).
        real: optional real samples to compute divergences (patch SWD) and compare distributions.
        cfg: EvalConfig
        save_json_path: if provided, writes results as JSON (vectors truncated).

    Returns:
        results: dict with metrics for generated (and real, if provided) and divergences.
    """
    if cfg is None:
        cfg = EvalConfig()

    g = _ensure_nhw(_to_numpy(generated))
    g = _normalize_to_unit(g)

    results: Dict[str, Any] = {"config": cfg.__dict__.copy()}

    # Binarize generated
    g_pore = binarize(g, threshold=cfg.threshold, fixed_threshold=cfg.fixed_threshold, invert=cfg.invert)

    # --- Generated metrics ---
    gen_stats: Dict[str, Any] = {}

    # Pore stats
    gen_stats["porosity"] = _mean_std(porosity(g_pore))
    try:
        cc = connected_components_stats(g_pore)
        gen_stats["num_components"] = _mean_std(cc["num_components"])
        gen_stats["mean_component_area"] = _mean_std(cc["mean_component_area"])
    except Exception as e:
        gen_stats["connected_components_error"] = str(e)

    try:
        gen_stats["specific_surface_2d"] = _mean_std(specific_surface_2d(g_pore))
    except Exception as e:
        gen_stats["specific_surface_error"] = str(e)

    try:
        chi = euler_characteristic_2d(g_pore)
        gen_stats["euler_characteristic"] = _mean_std(chi)
    except Exception as e:
        gen_stats["euler_error"] = str(e)

    # Minkowski
    try:
        mk = minkowski_functionals_2d(g_pore)
        gen_stats["minkowski"] = {
            "area_fraction": _mean_std(mk["area_fraction"]),
            "perimeter_density": _mean_std(mk["perimeter_density"]),
            "euler_density": _mean_std(mk["euler_density"]),
        }
    except Exception as e:
        gen_stats["minkowski_error"] = str(e)

    # Two-point correlation
    try:
        s2 = two_point_correlation(g_pore, nbins=cfg.radial_bins)
        gen_stats["two_point_correlation"] = {"r": s2["r"], "S2": s2["S2"]}
    except Exception as e:
        gen_stats["two_point_correlation_error"] = str(e)

    # Lineal path
    try:
        lp = lineal_path_function(g_pore, max_r=cfg.lineal_max_r, direction="both")
        gen_stats["lineal_path"] = lp
    except Exception as e:
        gen_stats["lineal_path_error"] = str(e)

    # Chord length
    try:
        cld = chord_length_distribution(g_pore, direction="both", max_len=cfg.chord_max_len)
        gen_stats["chord_length_distribution"] = cld
    except Exception as e:
        gen_stats["chord_length_error"] = str(e)

    # Pore size distribution
    try:
        psd = pore_size_distribution(g_pore, bins=cfg.psd_bins_count, max_radius=cfg.psd_max_radius)
        gen_stats["pore_size_distribution"] = psd
    except Exception as e:
        gen_stats["pore_size_error"] = str(e)

    # Power spectrum (use grayscale, not binary)
    try:
        ps = power_spectrum(g, nbins=cfg.psd_bins)
        gen_stats["power_spectrum"] = ps
    except Exception as e:
        gen_stats["power_spectrum_error"] = str(e)

    results["generated"] = gen_stats

    # --- Real metrics (optional) + divergences ---
    if real is not None:
        r = _ensure_nhw(_to_numpy(real))
        r = _normalize_to_unit(r)
        r_pore = binarize(r, threshold=cfg.threshold, fixed_threshold=cfg.fixed_threshold, invert=cfg.invert)

        real_stats: Dict[str, Any] = {}
        real_stats["porosity"] = _mean_std(porosity(r_pore))
        try:
            cc = connected_components_stats(r_pore)
            real_stats["num_components"] = _mean_std(cc["num_components"])
            real_stats["mean_component_area"] = _mean_std(cc["mean_component_area"])
        except Exception as e:
            real_stats["connected_components_error"] = str(e)

        try:
            real_stats["specific_surface_2d"] = _mean_std(specific_surface_2d(r_pore))
        except Exception as e:
            real_stats["specific_surface_error"] = str(e)

        try:
            chi = euler_characteristic_2d(r_pore)
            real_stats["euler_characteristic"] = _mean_std(chi)
        except Exception as e:
            real_stats["euler_error"] = str(e)

        try:
            mk = minkowski_functionals_2d(r_pore)
            real_stats["minkowski"] = {
                "area_fraction": _mean_std(mk["area_fraction"]),
                "perimeter_density": _mean_std(mk["perimeter_density"]),
                "euler_density": _mean_std(mk["euler_density"]),
            }
        except Exception as e:
            real_stats["minkowski_error"] = str(e)

        try:
            s2 = two_point_correlation(r_pore, nbins=cfg.radial_bins)
            real_stats["two_point_correlation"] = {"r": s2["r"], "S2": s2["S2"]}
        except Exception as e:
            real_stats["two_point_correlation_error"] = str(e)

        try:
            lp = lineal_path_function(r_pore, max_r=cfg.lineal_max_r, direction="both")
            real_stats["lineal_path"] = lp
        except Exception as e:
            real_stats["lineal_path_error"] = str(e)

        try:
            cld = chord_length_distribution(r_pore, direction="both", max_len=cfg.chord_max_len)
            real_stats["chord_length_distribution"] = cld
        except Exception as e:
            real_stats["chord_length_error"] = str(e)

        try:
            psd = pore_size_distribution(r_pore, bins=cfg.psd_bins_count, max_radius=cfg.psd_max_radius)
            real_stats["pore_size_distribution"] = psd
        except Exception as e:
            real_stats["pore_size_error"] = str(e)

        try:
            ps = power_spectrum(r, nbins=cfg.psd_bins)
            real_stats["power_spectrum"] = ps
        except Exception as e:
            real_stats["power_spectrum_error"] = str(e)

        results["real"] = real_stats

        # Divergences
        try:
            div = patch_distribution_divergence(
                gen_imgs=g,
                real_imgs=r,
                patch=cfg.patch,
                n_patches=cfg.n_patches,
                n_projections=cfg.swd_projections,
                seed=cfg.seed,
            )
            results["divergence"] = div
        except Exception as e:
            results["divergence_error"] = str(e)

    # Optional JSON save (truncate long vectors)
    if save_json_path is not None:
        serializable = {}
        for k, v in results.items():
            if isinstance(v, dict):
                serializable[k] = {}
                for kk, vv in v.items():
                    if isinstance(vv, dict):
                        serializable[k][kk] = _maybe_reduce_vectors(vv)
                    else:
                        serializable[k][kk] = vv
            else:
                serializable[k] = v

        def _convert(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.float32, np.float64)):
                return float(o)
            if isinstance(o, (np.int32, np.int64)):
                return int(o)
            return o

        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=_convert)

    return results
