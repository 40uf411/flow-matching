import torch
import numpy as np
from pathlib import Path
from torchvision.utils import save_image

from flow_matching.models import UNetModel
from flow_matching.solver import ODESolver, ModelWrapper

# =========================
# USER SETTINGS
# =========================
CKPT_PATH = "outputs/cfm/banderabrown_bin/ckpt_epoch_0150.pth"  # or ckpt.pth
OUTPUT_DIR = "sample_outputs"
N_SAMPLES = 256
H = 256
W = 256

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_STEPS = 101
STEP_SIZE = 0.05
METHOD = "midpoint"

# NEW: VRAM-SAFE SAMPLING
BATCH_SIZE = 64          # <- set this to fit your GPU (8/16/32 are typical)
AUTO_BATCH = True        # <- if True, will reduce batch size on CUDA OOM
MIN_BATCH_SIZE = 1

# Optional: speed/memory tweaks (safe defaults)
USE_AMP = True           # mixed precision can reduce VRAM (works best on recent GPUs)
TORCH_BACKENDS_BENCHMARK = True  # speed (may slightly increase memory)

# =========================
# Setup
# =========================
torch.manual_seed(SEED)
np.random.seed(SEED)

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = TORCH_BACKENDS_BENCHMARK

out = Path(OUTPUT_DIR)
out.mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE)

# =========================
# Load checkpoint (support both formats)
# =========================
ckpt = torch.load(CKPT_PATH, map_location="cpu")
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
else:
    state_dict = ckpt

# =========================
# Infer channels and class conditioning from checkpoint
# =========================
w_in = state_dict["input_blocks.0.0.weight"]   # [model_channels, in_channels, k, k]
C_IN = int(w_in.shape[1])

w_out = state_dict["out.2.weight"]             # [out_channels, model_channels, k, k]
C_OUT = int(w_out.shape[0])

if C_IN != C_OUT:
    raise RuntimeError(
        f"Checkpoint has C_IN={C_IN} and C_OUT={C_OUT}; expected them to match for image sampling."
    )

CLASS_COND = ("label_emb.weight" in state_dict)
if CLASS_COND:
    NUM_CLASSES = int(state_dict["label_emb.weight"].shape[0])
else:
    NUM_CLASSES = 1

DIM = (C_IN, H, W)
print(f"Inferred from checkpoint: DIM={DIM}, CLASS_COND={CLASS_COND}, NUM_CLASSES={NUM_CLASSES}")

# =========================
# Build model matching the checkpoint
# =========================
flow = UNetModel(
    dim=DIM,
    num_channels=64,
    num_res_blocks=2,
    class_cond=CLASS_COND,
    num_classes=NUM_CLASSES,
).to(DEVICE)

flow.load_state_dict(state_dict, strict=True)
flow.eval()
print("Checkpoint loaded:", CKPT_PATH)

# =========================
# Sampling wrapper
# =========================
class WrappedModel(ModelWrapper):
    def forward(self, x, t, **extras):
        return self.model(x=x, t=t, **extras)

solver = ODESolver(WrappedModel(flow))

# =========================
# Generate samples in mini-batches (VRAM-safe)
# =========================
time_grid = torch.linspace(0, 1, SAMPLE_STEPS, device=DEVICE)

# Preallocate on CPU to avoid growing lists + keep VRAM stable
samples_cpu = torch.empty((N_SAMPLES, *DIM), dtype=torch.float32, device="cpu")

# Optional: store labels too (helpful for debugging reproducibility)
labels_cpu = None
if CLASS_COND:
    labels_cpu = torch.empty((N_SAMPLES,), dtype=torch.long, device="cpu")

def sample_one_batch(bs: int):
    """Sample one batch on DEVICE and return CPU tensor."""
    x_init = torch.randn((bs, *DIM), device=DEVICE, dtype=torch.float32)
    if CLASS_COND:
        y = torch.randint(0, NUM_CLASSES, (bs,), device=DEVICE)
    else:
        y = None

    # autocast reduces VRAM; keep outputs float32 for saving
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (USE_AMP and DEVICE == "cuda") else nullcontext()

    with torch.no_grad():
        with amp_ctx:
            x = solver.sample(
                x_init=x_init,
                step_size=STEP_SIZE,
                method=METHOD,
                time_grid=time_grid,
                return_intermediates=False,
                y=y,
            )

    x = x.detach().float().cpu()
    y_cpu = y.detach().cpu() if CLASS_COND else None
    return x, y_cpu

# Python <3.10 compatibility: define nullcontext if needed
try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext:
        def __enter__(self): return None
        def __exit__(self, *args): return False

cur_bs = int(BATCH_SIZE)
i = 0
while i < N_SAMPLES:
    bs = min(cur_bs, N_SAMPLES - i)
    try:
        batch_x, batch_y = sample_one_batch(bs)
        samples_cpu[i : i + bs] = batch_x
        if CLASS_COND:
            labels_cpu[i : i + bs] = batch_y
        i += bs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"Sampled {i}/{N_SAMPLES} (batch={bs})")
    except RuntimeError as e:
        oom = ("out of memory" in str(e).lower()) and (DEVICE == "cuda")
        if (not oom) or (not AUTO_BATCH):
            raise
        # Back off batch size
        torch.cuda.empty_cache()
        new_bs = max(MIN_BATCH_SIZE, cur_bs // 2)
        if new_bs == cur_bs:
            raise
        cur_bs = new_bs
        print(f"[OOM] Reducing batch size to {cur_bs} and retrying...")

samples = samples_cpu
print("Generated samples:", tuple(samples.shape))

# =========================
# Save all samples as .npy
# =========================
npy_path = out / "generated_samples.npy"
np.save(npy_path, samples.numpy())
print("Saved:", npy_path)

# Optional: save labels if class-conditional
if CLASS_COND and labels_cpu is not None:
    labels_path = out / "generated_labels.npy"
    np.save(labels_path, labels_cpu.numpy())
    print("Saved:", labels_path)

# =========================
# Save 3 PNG previews
# =========================
for j in range(min(3, N_SAMPLES)):
    png_path = out / f"sample_{j:03d}.png"
    save_image(samples[j], png_path, normalize=False)
    print("Saved:", png_path)
