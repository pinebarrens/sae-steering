# SAE training loop driven by Hydra configs and cached activation arrays.

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from sae_steering.models.sae import TopKSAE


@dataclass
class TrainResult:
    final_checkpoint: Path
    steps: int
    d_input: int
    d_latent: int


class ActivationBatcher:
    # Random mini-batches from a 2D .npy activation matrix.

    def __init__(self, path: Path, device: torch.device):
        self.path = Path(path)
        self.array = np.load(self.path, mmap_mode="r")
        if self.array.ndim != 2:
            raise ValueError(f"expected 2D activations, got shape {self.array.shape}")
        self.n_rows, self.d_input = self.array.shape
        self.device = device

    def sample(self, batch_size: int, generator: torch.Generator) -> Tensor:
        idx = torch.randint(
            self.n_rows,
            (batch_size,),
            generator=generator,
            device="cpu",
        ).numpy()
        batch = np.asarray(self.array[idx], dtype=np.float32)
        return torch.from_numpy(batch).to(self.device)


class SAETrainer:
    # Small, explicit trainer for TopK SAEs.

    def __init__(
        self,
        sae: TopKSAE,
        activations_path: Path,
        out_dir: Path,
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        batch_size: int = 4096,
        n_steps: int = 25_000,
        log_every: int = 100,
        checkpoint_every: int = 5_000,
        device: str = "cuda",
        seed: int = 0,
        wandb_mode: str = "disabled",
        wandb_project: str = "sae-steering",
        run_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.device = torch.device(device)
        self.sae = sae.to(self.device)
        self.batcher = ActivationBatcher(Path(activations_path), self.device)
        if self.batcher.d_input != self.sae.d_input:
            raise ValueError(
                f"activation dim {self.batcher.d_input} does not match "
                f"SAE d_input {self.sae.d_input}"
            )

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.metadata = metadata or {}

        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        self._init_pre_encoder_bias()
        self.wandb = None
        if wandb_mode != "disabled":
            import wandb

            self.wandb = wandb.init(
                project=wandb_project,
                mode=wandb_mode,
                name=run_name,
                config=self._metadata_dict(),
            )

    def _metadata_dict(self) -> dict[str, Any]:
        return {
            "activation_path": str(self.batcher.path),
            "n_activations": self.batcher.n_rows,
            "d_input": self.sae.d_input,
            "d_latent": self.sae.d_latent,
            "k": self.sae.k,
            "dead_feature_threshold": self.sae.dead_feature_threshold,
            "aux_k": self.sae.aux_k,
            "aux_loss_weight": self.sae.aux_loss_weight,
            "batch_size": self.batch_size,
            "n_steps": self.n_steps,
            **self.metadata,
        }

    def _init_pre_encoder_bias(self) -> None:
        mean = np.mean(self.batcher.array, axis=0, dtype=np.float32)
        with torch.no_grad():
            self.sae.pre_encoder_bias.copy_(
                torch.from_numpy(mean).to(self.device)
            )
        logger.info("Initialized pre_encoder_bias to activation mean")

    def _metrics(
        self,
        x: Tensor,
        z: Tensor,
        x_hat: Tensor,
        aux_loss: Tensor,
    ) -> dict[str, float]:
        mse = torch.mean((x_hat - x) ** 2)
        centered = x - x.mean(dim=0, keepdim=True)
        total_var = torch.mean(centered**2).clamp_min(1e-12)
        explained_var = 1 - mse / total_var
        return {
            "mse": float(mse.detach().cpu()),
            "aux_loss": float(aux_loss.detach().cpu()),
            "l0": float(z.ne(0).sum(dim=1).float().mean().detach().cpu()),
            "explained_variance": float(explained_var.detach().cpu()),
            "dead_features": float(
                (self.sae.steps_since_active >= self.sae.dead_feature_threshold)
                .sum()
                .detach()
                .cpu()
            ),
        }

    def save_checkpoint(self, name: str, step: int) -> Path:
        path = self.out_dir / name
        torch.save(
            {
                "model_state_dict": self.sae.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": step,
                "metadata": self._metadata_dict(),
            },
            path,
        )
        return path

    def train(self) -> TrainResult:
        metadata_path = self.out_dir / "metadata.json"
        metadata_path.write_text(json.dumps(self._metadata_dict(), indent=2))

        self.sae.train()
        start = time.time()
        for step in range(1, self.n_steps + 1):
            x = self.batcher.sample(self.batch_size, self.rng)
            _, z, x_hat, aux_loss = self.sae(x)
            mse = torch.mean((x_hat - x) ** 2)
            loss = mse + aux_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.sae.normalize_decoder_columns()

            if step == 1 or step % self.log_every == 0:
                elapsed = max(time.time() - start, 1e-6)
                metrics = self._metrics(x, z, x_hat, aux_loss)
                metrics["loss"] = float(loss.detach().cpu())
                metrics["acts_per_second"] = step * self.batch_size / elapsed
                logger.info(
                    "step={step:,} loss={loss:.4g} mse={mse:.4g} "
                    "l0={l0:.1f} ev={ev:.3f} dead={dead:.0f} throughput={tput:.0f}/s",
                    step=step,
                    loss=metrics["loss"],
                    mse=metrics["mse"],
                    l0=metrics["l0"],
                    ev=metrics["explained_variance"],
                    dead=metrics["dead_features"],
                    tput=metrics["acts_per_second"],
                )
                if self.wandb is not None:
                    self.wandb.log(metrics, step=step)

            if step % self.checkpoint_every == 0:
                ckpt = self.save_checkpoint(f"step_{step:06d}.pt", step)
                logger.info(f"Saved checkpoint: {ckpt}")

        final = self.save_checkpoint("final.pt", self.n_steps)
        logger.info(f"Saved final checkpoint: {final}")
        if self.wandb is not None:
            self.wandb.finish()
        return TrainResult(
            final_checkpoint=final,
            steps=self.n_steps,
            d_input=self.sae.d_input,
            d_latent=self.sae.d_latent,
        )


def train(activations_path: Path, out_dir: Path, cfg) -> TrainResult:
    # Build a TopKSAE from config and train it on cached activations.
    sae = TopKSAE(
        d_input=int(cfg.sae.d_input),
        d_latent=int(cfg.sae.d_latent),
        k=int(cfg.sae.k),
        dead_feature_threshold=int(cfg.sae.dead_feature_threshold),
        aux_k=int(cfg.sae.aux_k),
        aux_loss_weight=float(cfg.sae.aux_loss_weight),
    )
    trainer = SAETrainer(
        sae,
        activations_path,
        out_dir,
        lr=float(cfg.optimizer.lr),
        betas=tuple(cfg.optimizer.betas),
        weight_decay=float(cfg.optimizer.weight_decay),
        batch_size=int(cfg.batch_size),
        n_steps=int(cfg.n_steps),
        log_every=int(cfg.log_every),
        checkpoint_every=int(cfg.checkpoint_every),
        device=str(cfg.device),
        seed=int(cfg.seed),
        wandb_mode=str(cfg.wandb.mode),
        wandb_project=str(cfg.wandb.project),
        metadata={"config": _to_plain_dict(cfg)},
    )
    return trainer.train()


def _to_plain_dict(obj: Any) -> Any:
    # Convert OmegaConf containers to JSON-friendly values.
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, DictConfig | ListConfig):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass

    if hasattr(obj, "items"):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_plain_dict(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj
