# Load trained SAE checkpoints and quality reports.

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from sae_steering.models.sae import TopKSAE

_QUALITY_THRESHOLDS = (0.01, 0.005, 0.001, 0.0005)


def load_sae_checkpoint(path: Path, device: str = "cpu") -> TopKSAE:
    # Reconstruct a TopKSAE from a final.pt checkpoint
    path = Path(path)
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    meta: dict[str, Any] = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}

    dec = state_dict["decoder.weight"]
    d_input = int(meta.get("d_input", dec.shape[0]))
    d_latent = int(meta.get("d_latent", dec.shape[1]))

    sae = TopKSAE(
        d_input=d_input,
        d_latent=d_latent,
        k=int(meta.get("k", 32)),
        dead_feature_threshold=int(meta.get("dead_feature_threshold", 10_000)),
        aux_k=int(meta.get("aux_k", 32)),
        aux_loss_weight=float(meta.get("aux_loss_weight", 1 / 32)),
    )
    sae.load_state_dict(state_dict)
    sae.to(device).eval()
    logger.info(
        f"Loaded SAE from {path} "
        f"(d_input={d_input}, d_latent={d_latent}, k={sae.k}, "
        f"step={ckpt.get('step') if isinstance(ckpt, dict) else '?'})"
    )
    return sae


@torch.no_grad()
def _firing_rates_batched(
    sae: TopKSAE,
    activations: np.ndarray,
    *,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, float]:
    # Return per-feature firing rates and mean MSE over all rows.
    device = next(sae.parameters()).device
    n_rows = int(activations.shape[0])
    firing_counts = torch.zeros(sae.d_latent, dtype=torch.float64)
    mse_sum = 0.0

    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        x = torch.as_tensor(
            np.asarray(activations[start:end], dtype=np.float32),
            device=device,
        )
        _, z, x_hat, _ = sae(x)
        firing_counts += (z != 0).double().sum(dim=0).cpu()
        mse_sum += torch.mean((x_hat - x) ** 2).item() * (end - start)

    firing_rate = firing_counts / n_rows
    return firing_rate, mse_sum / n_rows


@torch.no_grad()
def sae_quality_report(
    sae: TopKSAE,
    activations: np.ndarray,
    *,
    min_firing_rate: float = 0.001,
    min_live_features: int = 10,
    sample_size: int = 0,
    eval_batch_size: int = 4096,
) -> dict[str, Any]:
    # Compute reconstruction / sparsity diagnostics for an SAE
    x_np = activations if sample_size in (0, None) else activations[:sample_size]
    x_np = np.asarray(x_np)
    n_rows = int(x_np.shape[0])

    firing_rate, mse = _firing_rates_batched(
        sae, x_np, batch_size=eval_batch_size
    )
    x_mean = x_np.mean(axis=0, keepdims=True)
    total_var = float(np.mean((x_np - x_mean) ** 2).clip(min=1e-12))
    explained_variance = float(1 - mse / total_var)

    live = int((firing_rate >= min_firing_rate).sum())
    dead = int((firing_rate == 0).sum())
    mean_l0 = float(firing_rate.sum())

    live_at_threshold = {
        str(thr): int((firing_rate >= thr).sum()) for thr in _QUALITY_THRESHOLDS
    }
    active_ge_1_cell = int((firing_rate > 0).sum())
    active_ge_10_cells = int((firing_rate * n_rows >= 10).sum())

    report = {
        "n_activations": n_rows,
        "d_input": int(sae.d_input),
        "d_latent": int(sae.d_latent),
        "k": int(sae.k),
        "explained_variance": explained_variance,
        "explained_variance_in_sample": True,
        "live_feature_count": live,
        "dead_feature_count": dead,
        "mean_l0": mean_l0,
        "min_firing_rate": float(min_firing_rate),
        "live_at_threshold": live_at_threshold,
        "active_ge_1_cell": active_ge_1_cell,
        "active_ge_10_cells": active_ge_10_cells,
        "pass_quality": live >= min_live_features,
    }
    threshold_summary = " ".join(
        f"@{thr}={live_at_threshold[str(thr)]}" for thr in _QUALITY_THRESHOLDS
    )
    logger.info(
        f"SAE quality (n={n_rows}): EV(in-sample)={explained_variance:.3f} "
        f"live={live}/{sae.d_latent} dead={dead} mean_l0={mean_l0:.1f} "
        f"active>=1cell={active_ge_1_cell} active>=10cells={active_ge_10_cells} "
        f"{threshold_summary} pass={report['pass_quality']}"
    )
    return report
