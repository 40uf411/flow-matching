# discriminator.py
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------
def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: [B] tensor
        embedding_dim: embedding dimension
        max_period: controls minimum frequency

    Returns:
        [B, embedding_dim] tensor
    """
    if timesteps.dim() != 1:
        timesteps = timesteps.view(-1)

    half_dim = embedding_dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        / max(half_dim, 1)
    )
    args = timesteps[:, None].float() * frequencies[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))

    return embedding


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding followed by an MLP projection.
    """

    def __init__(
        self,
        embedding_dim: int,
        model_dim: int,
        max_period: int = 10000,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.max_period = max_period

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = get_timestep_embedding(
            timesteps=timesteps,
            embedding_dim=self.embedding_dim,
            max_period=self.max_period,
        )
        return self.mlp(emb)


class PlaneEmbedding(nn.Module):
    """
    Plane embedding:
      0 -> XY
      1 -> XZ
      2 -> YZ
    """

    def __init__(self, num_planes: int = 3, embedding_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_planes, embedding_dim)

    def forward(self, plane_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(plane_ids)


# ---------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------
class ConvBlock2D(nn.Module):
    """
    Downsampling conv block with FiLM-style conditioning from a shared cond vector.

    No batch norm / instance norm is used, which is usually safer for WGAN-GP.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        activation_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.act = nn.LeakyReLU(activation_slope, inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)

        gamma, beta = self.cond_proj(cond).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        h = h * (1.0 + gamma) + beta
        h = self.act(h)
        h = self.dropout(h)
        return h


@dataclass
class DiscriminatorOutput:
    score: torch.Tensor          # [B]
    features: torch.Tensor       # [B, C]


# ---------------------------------------------------------------------
# Main discriminator
# ---------------------------------------------------------------------
class ConditionalDiscriminator2D(nn.Module):
    """
    2D WGAN-GP discriminator conditioned on:
      - timestep embedding
      - plane embedding

    It consumes 2D slices of shape [B, C, H, W] and returns one scalar score per sample.

    Conditioning is used in two ways:
      1) FiLM-style modulation inside conv blocks
      2) projection-style conditional score at the pooled feature level
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        timestep_embedding_dim: int = 128,
        timestep_model_dim: int = 128,
        plane_embedding_dim: int = 128,
        cond_hidden_dim: int = 256,
        num_planes: int = 3,
        activation_slope: float = 0.2,
        dropout: float = 0.0,
        max_period: int = 10000,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.timestep_embedding_dim = timestep_embedding_dim
        self.timestep_model_dim = timestep_model_dim
        self.plane_embedding_dim = plane_embedding_dim
        self.cond_hidden_dim = cond_hidden_dim
        self.num_planes = num_planes

        # Conditioning modules
        self.time_embed = TimestepEmbedding(
            embedding_dim=timestep_embedding_dim,
            model_dim=timestep_model_dim,
            max_period=max_period,
        )
        self.plane_embed = PlaneEmbedding(
            num_planes=num_planes,
            embedding_dim=plane_embedding_dim,
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(timestep_model_dim + plane_embedding_dim, cond_hidden_dim),
            nn.SiLU(),
            nn.Linear(cond_hidden_dim, cond_hidden_dim),
        )

        # Feature extractor
        channels = [base_channels * m for m in channel_multipliers]

        self.input_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1)

        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(
                ConvBlock2D(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    cond_dim=cond_hidden_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    activation_slope=activation_slope,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.final_conv = nn.Conv2d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)
        self.final_act = nn.LeakyReLU(activation_slope, inplace=True)

        # Unconditional scalar head
        self.uncond_head = nn.Linear(channels[-1], 1)

        # Projection-style conditional head
        self.cond_proj = nn.Linear(cond_hidden_dim, channels[-1])

    def make_condition(
        self,
        timesteps: torch.Tensor,
        plane_ids: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embed(timesteps)      # [B, timestep_model_dim]
        p_emb = self.plane_embed(plane_ids)     # [B, plane_embedding_dim]
        cond = torch.cat([t_emb, p_emb], dim=1)
        cond = self.cond_mlp(cond)              # [B, cond_hidden_dim]
        return cond

    def forward_features(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        plane_ids: torch.Tensor,
    ) -> DiscriminatorOutput:
        """
        Args:
            x: [B, C, H, W]
            timesteps: [B]
            plane_ids: [B]

        Returns:
            DiscriminatorOutput(score=[B], features=[B, Cfeat])
        """
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape [B,C,H,W], got {tuple(x.shape)}")
        if timesteps.dim() != 1:
            timesteps = timesteps.view(-1)
        if plane_ids.dim() != 1:
            plane_ids = plane_ids.view(-1)

        cond = self.make_condition(timesteps, plane_ids)

        h = self.input_conv(x)
        h = F.leaky_relu(h, negative_slope=0.2, inplace=True)

        for block in self.blocks:
            h = block(h, cond)

        h = self.final_conv(h)
        h = self.final_act(h)

        # Global sum pooling is a common discriminator choice
        feat = h.sum(dim=(2, 3))  # [B, Cfeat]

        uncond_score = self.uncond_head(feat).squeeze(1)  # [B]

        cond_vec = self.cond_proj(cond)                   # [B, Cfeat]
        proj_score = (feat * cond_vec).sum(dim=1)         # [B]

        score = uncond_score + proj_score
        return DiscriminatorOutput(score=score, features=feat)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        plane_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_features(x, timesteps, plane_ids).score


# ---------------------------------------------------------------------
# WGAN-GP helper losses
# ---------------------------------------------------------------------
def discriminator_loss_wgan(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Wasserstein discriminator loss:
        L_D = E[fake] - E[real]
    """
    return fake_scores.mean() - real_scores.mean()


def generator_loss_wgan(
    fake_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Wasserstein generator loss:
        L_G = -E[fake]
    """
    return -fake_scores.mean()


def gradient_penalty(
    discriminator: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    timesteps: torch.Tensor,
    plane_ids: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty on interpolated samples.

    Args:
        discriminator: ConditionalDiscriminator2D
        real_samples: [B,C,H,W]
        fake_samples: [B,C,H,W]
        timesteps: [B]
        plane_ids: [B]
        lambda_gp: GP coefficient

    Returns:
        scalar penalty tensor
    """
    if real_samples.shape != fake_samples.shape:
        raise ValueError(
            f"real/fake shape mismatch: {tuple(real_samples.shape)} vs {tuple(fake_samples.shape)}"
        )

    device = real_samples.device
    batch_size = real_samples.size(0)

    alpha = torch.rand(batch_size, 1, 1, 1, device=device, dtype=real_samples.dtype)
    interpolates = alpha * real_samples + (1.0 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(
        interpolates,
        timesteps=timesteps,
        plane_ids=plane_ids,
    )

    grad_outputs = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    gp = ((grad_norm - 1.0) ** 2).mean()

    return lambda_gp * gp


# ---------------------------------------------------------------------
# Optional utility for extracting oriented 2D slices from 3D volumes
# ---------------------------------------------------------------------
def create_balanced_slice_indices(
    batch_size: int,
    slices_per_sample: int,
    max_slice_idx: int,
    device=None,
    strategy: str = "stratified",
) -> torch.Tensor:
    """
    Create slice indices of shape [B, K].

    Args:
        batch_size: number of 3D samples
        slices_per_sample: K
        max_slice_idx: number of valid slice positions along the chosen axis
        device: torch device
        strategy: 'random', 'uniform', 'stratified'

    Returns:
        [B, K] long tensor
    """
    if device is None:
        device = torch.device("cpu")

    if strategy == "random":
        return torch.randint(0, max_slice_idx, (batch_size, slices_per_sample), device=device)

    elif strategy == "uniform":
        base_indices = torch.linspace(0, max_slice_idx - 1, slices_per_sample, device=device).long()
        return base_indices.unsqueeze(0).expand(batch_size, -1)

    elif strategy == "stratified":
        if slices_per_sample > max_slice_idx:
            return torch.randint(0, max_slice_idx, (batch_size, slices_per_sample), device=device)

        stratum_size = max_slice_idx / slices_per_sample
        base_positions = torch.arange(
            slices_per_sample, device=device, dtype=torch.float32
        ) * stratum_size
        random_offsets = torch.rand(batch_size, slices_per_sample, device=device) * stratum_size
        indices = (base_positions.unsqueeze(0) + random_offsets).long()
        return torch.clamp(indices, max=max_slice_idx - 1)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def extract_oriented_slices(
    volumes: torch.Tensor,
    plane_ids: torch.Tensor,
    slices_per_sample: int = 1,
    indices: torch.Tensor | None = None,
    strategy: str = "stratified",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract K oriented 2D slices from each 3D volume.

    Args:
        volumes: [B, C, D, H, W]
        plane_ids: [B] where 0=XY, 1=XZ, 2=YZ
        slices_per_sample: number of slices to extract per 3D sample
        indices: optional [B, K] tensor of slice indices.
                 If None, indices are generated using `strategy`.
        strategy: 'random', 'uniform', 'stratified'

    Returns:
        slices: [B*K, C, H, W]
        plane_ids_out: [B*K]
        slice_indices_out: [B*K]
    """
    if volumes.dim() != 5:
        raise ValueError(f"Expected volumes [B,C,D,H,W], got {tuple(volumes.shape)}")

    B, C, D, H, W = volumes.shape
    device = volumes.device

    if plane_ids.dim() != 1 or plane_ids.shape[0] != B:
        raise ValueError(f"Expected plane_ids [B], got {tuple(plane_ids.shape)}")

    # Since your setup assumes cubic volumes, any of D/H/W could be used depending on plane.
    # We keep axis-specific limits for correctness.
    max_indices = torch.empty(B, device=device, dtype=torch.long)
    for i in range(B):
        pid = int(plane_ids[i].item())
        if pid == 0:      # XY -> choose z index
            max_indices[i] = D
        elif pid == 1:    # XZ -> choose y index
            max_indices[i] = H
        elif pid == 2:    # YZ -> choose x index
            max_indices[i] = W
        else:
            raise ValueError(f"Unknown plane id {pid}")

    if indices is None:
        # Generate per-sample indices. Since D==H==W in your current setup,
        # we can generate in one shot using D. For full generality, handle per-sample below.
        if torch.all(max_indices == max_indices[0]):
            indices = create_balanced_slice_indices(
                batch_size=B,
                slices_per_sample=slices_per_sample,
                max_slice_idx=int(max_indices[0].item()),
                device=device,
                strategy=strategy,
            )
        else:
            # Fallback if dimensions differ
            idx_list = []
            for i in range(B):
                idx_i = create_balanced_slice_indices(
                    batch_size=1,
                    slices_per_sample=slices_per_sample,
                    max_slice_idx=int(max_indices[i].item()),
                    device=device,
                    strategy=strategy,
                )
                idx_list.append(idx_i)
            indices = torch.cat(idx_list, dim=0)
    else:
        if indices.dim() != 2 or indices.shape != (B, slices_per_sample):
            raise ValueError(
                f"Expected indices shape {(B, slices_per_sample)}, got {tuple(indices.shape)}"
            )
        indices = indices.to(device=device, dtype=torch.long)

    out_slices = []
    out_plane_ids = []
    out_slice_indices = []

    for i in range(B):
        pid = int(plane_ids[i].item())
        idx_row = indices[i]  # [K]

        for idx in idx_row.tolist():
            if pid == 0:      # XY => fix z
                sl = volumes[i, :, idx, :, :]     # [C,H,W]
            elif pid == 1:    # XZ => fix y
                sl = volumes[i, :, :, idx, :]     # [C,D,W]
            elif pid == 2:    # YZ => fix x
                sl = volumes[i, :, :, :, idx]     # [C,D,H]
            else:
                raise ValueError(f"Unknown plane id {pid}")

            out_slices.append(sl)
            out_plane_ids.append(pid)
            out_slice_indices.append(idx)

    slices = torch.stack(out_slices, dim=0)  # [B*K,C,H,W]
    plane_ids_out = torch.tensor(out_plane_ids, device=device, dtype=torch.long)
    slice_indices_out = torch.tensor(out_slice_indices, device=device, dtype=torch.long)

    return slices, plane_ids_out, slice_indices_out