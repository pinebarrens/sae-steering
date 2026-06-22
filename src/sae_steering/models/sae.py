# TopK sparse autoencoder.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


@dataclass
class SAEConfig:
    # Compatibility config for the older SparseAutoencoder name.

    d_in: int = 512
    d_sae: int = 8192
    k: int = 32
    dead_feature_threshold: int = 10_000
    aux_k: int = 32
    aux_loss_weight: float = 1 / 32


class TopKSAE(nn.Module):
    # TopK sparse autoencoder for residual-stream activations

    def __init__(
        self,
        d_input: int = 512,
        d_latent: int = 8192,
        k: int = 32,
        dead_feature_threshold: int = 10_000,
        aux_k: int = 32,
        aux_loss_weight: float = 1 / 32,
    ):
        super().__init__()
        if not 0 < k <= d_latent:
            raise ValueError(f"k must be in [1, d_latent], got k={k}")

        self.d_input = d_input
        self.d_latent = d_latent
        self.k = k
        self.dead_feature_threshold = dead_feature_threshold
        self.aux_k = min(aux_k, d_latent)
        self.aux_loss_weight = aux_loss_weight

        self.encoder = nn.Linear(d_input, d_latent)
        self.decoder = nn.Linear(d_latent, d_input, bias=False)
        self.pre_encoder_bias = nn.Parameter(torch.zeros(d_input))

        self.register_buffer(
            "steps_since_active",
            torch.zeros(d_latent, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("step", torch.zeros((), dtype=torch.long), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Kaiming init, then tied decoder init and unit norms.
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            self.encoder.weight.div_(
                self.encoder.weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
            )
            self.decoder.weight.copy_(self.encoder.weight.T)
            self.normalize_decoder_columns()

    def encode(
        self, x: Float[Tensor, "batch d_input"]
    ) -> tuple[Float[Tensor, "batch d_latent"], Float[Tensor, "batch d_latent"]]:
        # Return dense pre-activations and exact TopK sparse activations.
        z_pre = self.encoder(x - self.pre_encoder_bias)
        topk_vals, topk_idx = z_pre.topk(self.k, dim=-1)
        z = torch.zeros_like(z_pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return z_pre, z

    def decode(
        self, z: Float[Tensor, "batch d_latent"]
    ) -> Float[Tensor, "batch d_input"]:
        return self.decoder(z) + self.pre_encoder_bias

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        # Project every decoder feature direction back to unit L2 norm.
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.div_(norms)

    @torch.no_grad()
    def _update_activity(self, z: Tensor) -> None:
        active = z.ne(0).any(dim=0)
        self.steps_since_active.add_(1)
        self.steps_since_active[active] = 0
        self.step.add_(1)

    def _dead_feature_aux_loss(
        self,
        x: Tensor,
        z_pre: Tensor,
        x_hat: Tensor,
    ) -> Tensor:
        # Reconstruct residuals using currently dead features only.
        dead = self.steps_since_active >= self.dead_feature_threshold
        if self.aux_k == 0 or not bool(dead.any()):
            return x_hat.new_zeros(())

        dead_idx = dead.nonzero(as_tuple=False).flatten()
        k_aux = min(self.aux_k, int(dead_idx.numel()))
        dead_scores = z_pre[:, dead_idx]
        top_vals, top_pos = dead_scores.topk(k_aux, dim=-1)

        z_aux = torch.zeros_like(z_pre)
        aux_idx = dead_idx[top_pos]
        z_aux.scatter_(-1, aux_idx, top_vals)

        residual = (x - x_hat).detach()
        aux_recon = self.decoder(z_aux)
        return self.aux_loss_weight * torch.mean((aux_recon - residual) ** 2)

    def forward(
        self, x: Float[Tensor, "batch d_input"]
    ) -> tuple[
        Float[Tensor, "batch d_latent"],
        Float[Tensor, "batch d_latent"],
        Float[Tensor, "batch d_input"],
        Tensor,
    ]:
        z_pre, z = self.encode(x)
        x_hat = self.decode(z)
        aux_loss = self._dead_feature_aux_loss(x, z_pre, x_hat)
        if self.training:
            self._update_activity(z.detach())
        return z_pre, z, x_hat, aux_loss


class SparseAutoencoder(TopKSAE):
    # Backward-compatible wrapper around TopKSAE.

    def __init__(self, cfg: SAEConfig):
        super().__init__(
            d_input=cfg.d_in,
            d_latent=cfg.d_sae,
            k=cfg.k,
            dead_feature_threshold=cfg.dead_feature_threshold,
            aux_k=cfg.aux_k,
            aux_loss_weight=cfg.aux_loss_weight,
        )
        self.cfg = cfg

    def forward(
        self, x: Float[Tensor, "batch d_input"]
    ) -> tuple[Float[Tensor, "batch d_input"], Float[Tensor, "batch d_latent"]]:
        _, z, x_hat, _ = super().forward(x)
        return x_hat, z
