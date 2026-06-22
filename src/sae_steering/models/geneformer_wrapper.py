# Geneformer V2-104M activation extraction (mirrors scGPT wrapper API).


from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from anndata import AnnData
from loguru import logger

if TYPE_CHECKING:
    from datasets import Dataset


# Geneformer V2 special token ids (from token_dictionary_gc104M.pkl)
CLS_ID = 2
EOS_ID = 3
PAD_ID = 0

# V2-104M architecture constants (from config.json)
N_LAYERS = 12
D_MODEL = 768
MAX_POS = 4096


class GeneformerActivationExtractor:
    # Load Geneformer V2-104M and extract layer activations

    def __init__(
        self,
        checkpoint_dir: Path,
        layer: int = 6,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        pool: Literal["mean", "cls"] = "mean",
        gene_median_path: Path | None = None,
        token_dict_path: Path | None = None,
        tmp_root: Path | None = None,
        nproc: int | None = None,
    ):
        try:
            from geneformer import TranscriptomeTokenizer  # noqa: F401
        except ImportError as e:
            raise NotImplementedError(
                "geneformer package not installed. "
                "Run `make download-geneformer` then `make install-geneformer-pkg`."
            ) from e

        from transformers import BertForMaskedLM

        self.checkpoint_dir = Path(checkpoint_dir)
        self.layer = layer
        self.device = torch.device(device)
        self.dtype = dtype
        self.pool = pool

        # Resolve V2-104M-specific tokenizer assets
        self.gene_median_path = self._resolve_asset(
            gene_median_path, "gene_median_dictionary_gc104M.pkl"
        )
        self.token_dict_path = self._resolve_asset(
            token_dict_path, "token_dictionary_gc104M.pkl"
        )
        logger.info(
            f"Geneformer assets: gene_median={self.gene_median_path}, "
            f"token_dict={self.token_dict_path}"
        )

        self.tmp_root = Path(tmp_root) if tmp_root is not None else None
        if nproc is None:
            nproc = min(4, os.cpu_count() or 1)
        self.nproc = nproc

        logger.info(f"Loading Geneformer V2-104M from {self.checkpoint_dir}")
        self.model = BertForMaskedLM.from_pretrained(
            self.checkpoint_dir,
            output_hidden_states=True,
            output_attentions=False,
            attn_implementation="sdpa",
        )

        cfg = self.model.config
        assert cfg.num_hidden_layers == N_LAYERS, (
            f"expected {N_LAYERS} layers, got {cfg.num_hidden_layers}"
        )
        assert cfg.hidden_size == D_MODEL, (
            f"expected hidden_size={D_MODEL}, got {cfg.hidden_size}"
        )
        assert cfg.pad_token_id == PAD_ID, (
            f"expected pad_token_id={PAD_ID}, got {cfg.pad_token_id}"
        )

        self.model.to(self.device).eval()

    def _resolve_asset(self, user_path: Path | None, default_name: str) -> Path | None:
        if user_path is not None:
            return Path(user_path)
        local = self.checkpoint_dir / default_name
        if local.exists():
            return local
        return None

    def preprocess(self, adata: AnnData) -> Dataset:
        # Tokenize an AnnData of raw counts into a HF Dataset of input_ids
        import tempfile

        from datasets import load_from_disk
        from geneformer import TranscriptomeTokenizer

        # Copy so the caller's object is never mutated
        adata = adata.copy()

        X = adata.X
        raw = X.data if hasattr(X, "data") else X
        sample = np.asarray(raw).ravel()[:1000]
        if len(sample) > 0 and not np.allclose(sample % 1, 0):
            logger.warning(
                "adata.X does not look like raw integer counts "
                f"(max sample value={float(sample.max()):.3f}); Geneformer's "
                "rank-value encoding will be meaningless on normalized data."
            )

        # n_counts is required by the tokenizer and crashes if missing
        if "n_counts" not in adata.obs.columns:
            row_sums = adata.X.sum(axis=1)
            if hasattr(row_sums, "A1"):
                row_sums = row_sums.A1
            n_counts = np.asarray(row_sums).ravel()
            adata.obs["n_counts"] = n_counts
            logger.info("Computed adata.obs['n_counts'] from raw counts")
            assert (n_counts > 0).all(), "found cells with n_counts == 0"

        if "ensembl_id" not in adata.var.columns:
            if "feature_id" in adata.var.columns:
                adata.var["ensembl_id"] = adata.var["feature_id"].astype(str).values
                logger.info("Copied adata.var['feature_id'] -> adata.var['ensembl_id']")
            else:
                adata.var["ensembl_id"] = adata.var.index.astype(str).values
                logger.info("Copied adata.var.index -> adata.var['ensembl_id']")

        # Strip Ensembl version suffixes
        adata.var["ensembl_id"] = (
            adata.var["ensembl_id"].astype(str).str.split(".").str[0]
        )

        with tempfile.TemporaryDirectory(
            dir=str(self.tmp_root) if self.tmp_root else None
        ) as tmp:
            tmp = Path(tmp)
            in_dir = tmp / "in"
            out_dir = tmp / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            adata.write_h5ad(in_dir / "cells.h5ad")

            tokenizer_kwargs: dict = dict(
                custom_attr_name_dict=None,
                nproc=self.nproc,
                model_version="V2",
            )
            if self.gene_median_path is not None:
                tokenizer_kwargs["gene_median_file"] = str(self.gene_median_path)
            if self.token_dict_path is not None:
                tokenizer_kwargs["token_dictionary_file"] = str(self.token_dict_path)

            tk = TranscriptomeTokenizer(**tokenizer_kwargs)
            tk.tokenize_data(
                data_directory=str(in_dir),
                output_directory=str(out_dir),
                output_prefix="cells",
                file_format="h5ad",
            )
            dataset = load_from_disk(str(out_dir / "cells.dataset"))

        assert "input_ids" in dataset.column_names, (
            f"tokenizer output missing 'input_ids'; got {dataset.column_names}"
        )
        logger.info(
            f"Tokenized {len(dataset)} cells; columns = {dataset.column_names}"
        )
        return dataset

    @torch.no_grad()
    def extract_activations(
        self,
        data: AnnData | Dataset,
        batch_size: int = 32,
        capture_token_level: bool = False,
    ) -> np.ndarray:
        # Run the model and return residual-stream activations from `self.layer`
        if isinstance(data, AnnData):
            dataset = self.preprocess(data)
        else:
            dataset = data

        all_input_ids: list[list[int]] = list(dataset["input_ids"])
        n_cells = len(all_input_ids)
        max_len_in_dataset = max(len(ids) for ids in all_input_ids)
        logger.info(
            f"Dataset has {n_cells} cells; max sequence length = {max_len_in_dataset}"
        )
        if max_len_in_dataset < MAX_POS and max_len_in_dataset in (2048,):
            logger.warning(
                f"max sequence length {max_len_in_dataset} suggests the "
                "tokenizer truncated at 2048 (older Geneformer default); "
                "V2-104M expects 4096."
            )

        dataset_max_len = max_len_in_dataset if capture_token_level else 0

        pooled_chunks: list[np.ndarray] = []
        token_chunks: list[np.ndarray] = []

        autocast_enabled = self.dtype in (torch.float16, torch.bfloat16)
        if autocast_enabled:
            amp_ctx = torch.autocast(device_type=self.device.type, dtype=self.dtype)
        else:
            amp_ctx = contextlib.nullcontext()

        target_hidden_idx = self.layer + 1

        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            batch = all_input_ids[start:end]
            batch_max_len = max(len(ids) for ids in batch)

            # Right-pad to per-batch max length with PAD_ID=0
            input_ids = torch.full(
                (len(batch), batch_max_len), PAD_ID, dtype=torch.long
            )
            for i, ids in enumerate(batch):
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask = (input_ids != PAD_ID).long()

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            with amp_ctx:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            hidden = outputs.hidden_states[target_hidden_idx].float()  # (B, T, 768)

            if capture_token_level:
                pad_amount = dataset_max_len - hidden.shape[1]
                if pad_amount > 0:
                    pad_block = torch.zeros(
                        hidden.shape[0],
                        pad_amount,
                        hidden.shape[2],
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                    hidden = torch.cat([hidden, pad_block], dim=1)
                token_chunks.append(hidden.cpu().numpy())
            else:
                if self.pool == "cls":
                    pooled = hidden[:, 0, :]
                else:
                    # Mean over gene-token positions only: exclude PAD, CLS, and EOS
                    pool_mask = (
                        (input_ids != PAD_ID)
                        & (input_ids != CLS_ID)
                        & (input_ids != EOS_ID)
                    ).float().unsqueeze(-1)  # (B, T, 1)
                    pooled = (hidden * pool_mask).sum(dim=1) / pool_mask.sum(
                        dim=1
                    ).clamp(min=1)
                pooled_chunks.append(pooled.cpu().numpy())

            del hidden, outputs

        if capture_token_level:
            return np.concatenate(token_chunks, axis=0)
        return np.concatenate(pooled_chunks, axis=0)
