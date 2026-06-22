# scGPT checkpoint loading and residual-stream activation extraction.


from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from loguru import logger
from torch import Tensor

from sae_steering.data.gene_mapping import load_mapping, var_ensembl_ids

_EXTRACTION_SEED = 0


def _set_deterministic_seed(seed: int = _EXTRACTION_SEED) -> None:
    # Fix preprocess + forward RNG for reproducible activations.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_dense_hidden(hidden: Tensor) -> Tensor:
    # Convert layer hidden states to dense (B, L, d_model)
    if getattr(hidden, "is_nested", False):
        return hidden.to_padded_tensor(0.0)
    return hidden


class scGPTActivationExtractor:
    # Load scGPT checkpoint and extract layer activations

    def __init__(
        self,
        checkpoint_dir: Path,
        gene_mapping_path: Path,
        layer: int = 6,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        max_seq_len: int = 1200,
        n_hvg: int = 1199,
        n_bins: int = 51,
        pool: Literal["mean", "cls"] = "mean",
    ):
        from scgpt.model import TransformerModel
        from scgpt.tokenizer.gene_tokenizer import GeneVocab

        self.checkpoint_dir = Path(checkpoint_dir)
        self.layer = layer
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.n_hvg = n_hvg
        self.n_bins = n_bins
        self.pool = pool

        # Load training-time hyperparameters; these define the model architecture.
        args_path = self.checkpoint_dir / "args.json"
        with open(args_path) as f:
            self.args = json.load(f)
        logger.info(f"Loaded args.json from {args_path}")

        # GeneVocab maps HGNC symbol -> integer id
        vocab_path = self.checkpoint_dir / "vocab.json"
        self.vocab = GeneVocab.from_file(vocab_path)
        self.vocab.set_default_index(self.vocab["<pad>"])
        self.pad_token = "<pad>"
        self.pad_value = -2
        logger.info(f"Loaded vocab ({len(self.vocab)} tokens) from {vocab_path}")

        # Try to use flash-attn if installed
        try:
            import flash_attn  # noqa: F401

            fast_transformer = bool(self.args.get("fast_transformer", True))
            logger.info("flash-attn detected; using fast transformer backend")
        except ImportError:
            fast_transformer = False
            logger.info("flash-attn not available; using vanilla pytorch transformer")
        self._use_fast_transformer = fast_transformer

        self._trained_no_cls = bool(self.args.get("no_cls", False))
        cell_emb_style = "avg-pool" if self._trained_no_cls else "cls"

        # Build the model with kwargs matching args.json
        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.args["embsize"],
            nhead=self.args["nheads"],
            d_hid=self.args["d_hid"],
            nlayers=self.args["nlayers"],
            nlayers_cls=self.args.get("n_layers_cls", 3),
            n_cls=1,
            vocab=self.vocab,
            dropout=self.args["dropout"],
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            input_emb_style=self.args.get("input_emb_style", "continuous"),
            n_input_bins=self.args.get("n_bins", n_bins),
            cell_emb_style=cell_emb_style,
            mvc_decoder_style="inner product",
            ecs_threshold=0.0,
            explicit_zero_prob=False,
            use_fast_transformer=fast_transformer,
            fast_transformer_backend="flash",
            pre_norm=False,
        )

        ckpt_path = self.checkpoint_dir / "best_model.pt"
        state_dict = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        logger.info(
            f"Loaded {ckpt_path.name}: "
            f"{len(missing)} missing keys, {len(unexpected)} unexpected keys"
        )

        self.model.to(self.device).eval()

        self.gene_mapping = load_mapping(Path(gene_mapping_path))
        self.ensembl_to_symbol = dict(
            zip(self.gene_mapping["ensembl_id"], self.gene_mapping["symbol"], strict=True)
        )
        logger.info(f"Loaded gene mapping ({len(self.gene_mapping)} entries)")

        # Mixed-cohort preprocess fitted once
        self._preprocess_reference: AnnData | None = None
        self._preprocess_reference_obs: set[str] | None = None

    def _forward_padding_mask(self, src_key_padding_mask: Tensor) -> Tensor | None:
        # Pass padding mask only for flash-attn
        return src_key_padding_mask if self._use_fast_transformer else None

    def _map_to_vocab(self, adata: AnnData) -> AnnData:
        # Ensembl -> symbol, drop unmapped/OOV genes
        ensembl_ids = var_ensembl_ids(adata)
        symbols = pd.Series(
            [self.ensembl_to_symbol.get(eid) for eid in ensembl_ids],
            index=adata.var.index,
        )
        keep_mapped = symbols.notna().values
        n_unmapped = int((~keep_mapped).sum())
        adata = adata[:, keep_mapped].copy()
        adata.var["gene_name"] = symbols[keep_mapped].values
        logger.info(
            f"Resolved Ensembl -> symbol: dropped {n_unmapped} unmapped genes, "
            f"{adata.n_vars} remain"
        )

        vocab_tokens = set(self.vocab.get_itos())
        in_vocab = adata.var["gene_name"].isin(vocab_tokens).values
        n_oov = int((~in_vocab).sum())
        adata = adata[:, in_vocab].copy()
        logger.info(
            f"Filtered to scGPT vocab: dropped {n_oov} OOV genes, "
            f"{adata.n_vars} remain"
        )
        return adata

    def _run_scgpt_preprocessor(self, adata: AnnData) -> AnnData:
        # Normalize -> log1p -> HVG -> binning via scGPT's Preprocessor.
        from scgpt.preprocess import Preprocessor

        preprocessor = Preprocessor(
            use_key="X",
            filter_gene_by_counts=False,
            filter_cell_by_counts=False,
            normalize_total=1e4,
            result_normed_key="X_normed",
            log1p=True,
            result_log1p_key="X_log1p",
            subset_hvg=self.n_hvg,
            hvg_flavor="seurat_v3",
            binning=self.n_bins,
            result_binned_key="X_binned",
        )
        preprocessor(adata, batch_key=None)
        logger.info(
            f"Preprocessed: {adata.n_obs} cells x {adata.n_vars} HVGs, "
            f"binned with {self.n_bins} bins"
        )
        return adata

    def fit_preprocess_reference(self, adata: AnnData) -> AnnData:
        # Preprocess a mixed cohort once; later preprocess() subsets from it
        mapped = self._map_to_vocab(adata)
        processed = self._run_scgpt_preprocessor(mapped)
        self._preprocess_reference = processed
        self._preprocess_reference_obs = set(processed.obs_names.astype(str))
        logger.info(
            f"Fitted preprocess reference: {processed.n_obs} cells x "
            f"{processed.n_vars} HVGs"
        )
        return processed

    def clear_preprocess_reference(self) -> None:
        # Drop the cached mixed-cohort preprocess (e.g
        self._preprocess_reference = None
        self._preprocess_reference_obs = None

    def preprocess(self, adata: AnnData) -> AnnData:
        if self._preprocess_reference is not None and self._preprocess_reference_obs is not None:
            names = adata.obs_names.astype(str)
            missing = set(names) - self._preprocess_reference_obs
            if missing:
                raise ValueError(
                    f"{len(missing)} cell(s) not in preprocess reference; "
                    "call fit_preprocess_reference on the parent AnnData first"
                )
            if len(names) == self._preprocess_reference.n_obs and set(names) == self._preprocess_reference_obs:
                return self._preprocess_reference
            return self._preprocess_reference[names].copy()

        mapped = self._map_to_vocab(adata)
        return self._run_scgpt_preprocessor(mapped)

    def tokenize(self, adata: AnnData) -> dict[str, Tensor]:
        # Build the (gene_id, binned_value) token pair tensors that scGPT expects
        from scgpt.tokenizer.gene_tokenizer import tokenize_and_pad_batch

        # The binning step can leave a sparse layer behind
        binned = adata.layers["X_binned"]
        if hasattr(binned, "toarray"):
            binned = binned.toarray()
        values = binned.astype(np.float32)

        # vocab(...) returns int ids; cast to int64 for torch.
        gene_ids = np.array(
            self.vocab(adata.var["gene_name"].tolist()), dtype=np.int64
        )

        tokenized = tokenize_and_pad_batch(
            data=values,
            gene_ids=gene_ids,
            max_len=self.max_seq_len,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            append_cls=True,  # function prepends <cls> at position 0
            include_zero_gene=False,
            cls_token="<cls>",
            return_pt=True,
        )
        return {
            "genes": tokenized["genes"].long(),
            "values": tokenized["values"].float(),
        }

    @torch.no_grad()
    def extract_activations(
        self,
        adata: AnnData,
        batch_size: int = 32,
        capture_token_level: bool = False,
    ) -> np.ndarray:
        # Run the model and return residual-stream activations from `self.layer`
        if self.pool == "cls" and self._trained_no_cls:
            logger.warning(
                "pool='cls' but checkpoint was trained with no_cls=True; "
                "position 0 was not a dedicated cell summary during training."
            )

        _set_deterministic_seed()
        processed = self.preprocess(adata)
        tokens = self.tokenize(processed)
        genes_all = tokens["genes"]
        values_all = tokens["values"]
        n_cells = genes_all.shape[0]
        pad_id = self.vocab[self.pad_token]

        captured: dict[str, Tensor] = {}

        def hook(module, inputs, output):
            captured["h"] = output if isinstance(output, Tensor) else output[0]

        handle = self.model.transformer_encoder.layers[self.layer].register_forward_hook(
            hook
        )

        pooled_chunks: list[np.ndarray] = []
        token_chunks: list[np.ndarray] = []

        # Mixed precision context for fp16/bf16 dtypes
        autocast_enabled = self.dtype in (torch.float16, torch.bfloat16)
        if autocast_enabled:
            amp_ctx = torch.autocast(device_type=self.device.type, dtype=self.dtype)
        else:
            amp_ctx = contextlib.nullcontext()

        try:
            for start in range(0, n_cells, batch_size):
                end = min(start + batch_size, n_cells)
                src = genes_all[start:end].to(self.device)
                values = values_all[start:end].to(self.device)
                # PyTorch convention: True = position is padding, ignore it.
                src_key_padding_mask = src.eq(pad_id)

                with amp_ctx:
                    _ = self.model(
                        src=src,
                        values=values,
                        src_key_padding_mask=self._forward_padding_mask(
                            src_key_padding_mask
                        ),
                        batch_labels=None,
                        CLS=False,  # cls_decoder has no checkpoint weights; skip
                        CCE=False,
                        MVC=False,
                        ECS=False,
                    )

                hidden = _to_dense_hidden(captured["h"].float())  # (B, L, d_model)

                if capture_token_level:
                    token_chunks.append(hidden.cpu().numpy())
                else:
                    if self.pool == "cls":
                        pooled = hidden[:, 0, :]
                    else:
                        # Masked mean over non-pad GENE positions
                        mask = (~src_key_padding_mask).float().unsqueeze(-1)  # (B, L, 1)
                        mask[:, 0, :] = 0.0
                        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

                    pooled_chunks.append(pooled.cpu().numpy())

                del hidden
                captured.clear()
        finally:
            handle.remove()
            captured.clear()

        if capture_token_level:
            return np.concatenate(token_chunks, axis=0)
        return np.concatenate(pooled_chunks, axis=0)

    @torch.no_grad()
    def extract_with_gepc(
        self,
        adata: AnnData,
        batch_size: int = 32,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.pool == "cls" and self._trained_no_cls:
            logger.warning(
                "pool='cls' but checkpoint was trained with no_cls=True; "
                "position 0 was not a dedicated cell summary during training."
            )

        _set_deterministic_seed()
        processed = self.preprocess(adata)
        tokens = self.tokenize(processed)
        genes_all = tokens["genes"]
        values_all = tokens["values"]
        n_cells = genes_all.shape[0]
        pad_id = self.vocab[self.pad_token]

        captured: dict[str, Tensor] = {}

        def hook(module, inputs, output):
            captured["h"] = output if isinstance(output, Tensor) else output[0]

        handle = self.model.transformer_encoder.layers[self.layer].register_forward_hook(
            hook
        )

        pooled_chunks: list[np.ndarray] = []
        gepc_chunks: list[np.ndarray] = []

        autocast_enabled = self.dtype in (torch.float16, torch.bfloat16)
        if autocast_enabled:
            amp_ctx = torch.autocast(device_type=self.device.type, dtype=self.dtype)
        else:
            amp_ctx = contextlib.nullcontext()

        try:
            for start in range(0, n_cells, batch_size):
                end = min(start + batch_size, n_cells)
                src = genes_all[start:end].to(self.device)
                values = values_all[start:end].to(self.device)
                src_key_padding_mask = src.eq(pad_id)

                with amp_ctx:
                    output_dict = self.model(
                        src=src,
                        values=values,
                        src_key_padding_mask=self._forward_padding_mask(
                            src_key_padding_mask
                        ),
                        batch_labels=None,
                        CLS=False,
                        CCE=False,
                        MVC=True,  # turn on gene-expression prediction head
                        ECS=False,
                    )

                hidden = _to_dense_hidden(captured["h"].float())  # (B, L, d_model)
                # MVC output is per-token (per-gene-in-sequence), shape (B, L)
                mvc_output = output_dict["mvc_output"].float()  # (B, L)
                gepc = mvc_output[:, 1:]  # (B, L-1) == (B, n_input_genes_padded)

                if self.pool == "cls":
                    pooled = hidden[:, 0, :]
                else:
                    # See extract_activations for why CLS is excluded.
                    mask = (~src_key_padding_mask).float().unsqueeze(-1)
                    mask[:, 0, :] = 0.0
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

                pooled_chunks.append(pooled.cpu().numpy())
                gepc_chunks.append(gepc.cpu().numpy())

                del hidden
                captured.clear()
        finally:
            handle.remove()
            captured.clear()

        pooled_all = np.concatenate(pooled_chunks, axis=0)
        gepc_all = np.concatenate(gepc_chunks, axis=0)
        return pooled_all, gepc_all

    def _resolve_steering_mode(
        self,
        steering_mode: Literal["auto", "gene", "cls", "gene_and_cls"],
    ) -> str:
        # Map "auto" to the mode that matches the checkpoint's pooling
        if steering_mode == "auto":
            return "gene" if self._trained_no_cls else "cls"
        if steering_mode not in ("gene", "cls", "gene_and_cls"):
            raise ValueError(
                f"unknown steering_mode {steering_mode!r}; "
                "expected one of auto|gene|cls|gene_and_cls"
            )
        return steering_mode

    @staticmethod
    def _steering_mask(src_key_padding_mask: Tensor, mode: str) -> Tensor:
        # Build the (B, L, 1) float mask of positions to add the offset to
        valid = (~src_key_padding_mask).float().unsqueeze(-1)  # (B, L, 1); pad -> 0
        if mode == "gene":
            valid[:, 0, :] = 0.0  # exclude CLS
        elif mode == "cls":
            cls = torch.zeros_like(valid)
            cls[:, 0, :] = 1.0
            valid = cls
        elif mode == "gene_and_cls":
            pass  # CLS is never pad, so it is already 1 here
        else:
            raise ValueError(f"unknown steering_mode {mode!r}")
        return valid

    @torch.no_grad()
    def forward_with_steering(
        self,
        adata: AnnData,
        offset: np.ndarray | Tensor,
        alpha: float,
        batch_size: int = 32,
        *,
        return_genes: bool = False,
        return_pooled: bool = False,
        steering_mode: Literal["auto", "gene", "cls", "gene_and_cls"] = "auto",
        pool_layer: int | None = None,
        tokens: dict[str, Tensor] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        _set_deterministic_seed()
        if tokens is None:
            processed = self.preprocess(adata)
            tokens = self.tokenize(processed)
        genes_all = tokens["genes"]
        values_all = tokens["values"]
        n_cells = genes_all.shape[0]
        pad_id = self.vocab[self.pad_token]
        offset_t = torch.as_tensor(
            np.asarray(offset), dtype=torch.float32, device=self.device
        )

        mode = self._resolve_steering_mode(steering_mode)
        pool_layer_idx = self.layer if pool_layer is None else pool_layer
        pool_at_hook = pool_layer_idx == self.layer

        state: dict[str, Tensor] = {}
        captured: dict[str, Tensor] = {}

        def steer_hook(module, inputs, output):
            is_tuple = not isinstance(output, Tensor)
            hidden = output[0] if is_tuple else output
            if alpha != 0.0:
                hidden = hidden + alpha * state["steer_mask"] * offset_t
            if return_pooled and pool_at_hook:
                captured["pool"] = hidden  # post-steer hidden at the SAE layer
            if is_tuple:
                return (hidden, *output[1:])
            return hidden

        handles = [
            self.model.transformer_encoder.layers[self.layer].register_forward_hook(
                steer_hook
            )
        ]
        if return_pooled and not pool_at_hook:
            # Pooling at a different layer (e.g
            def pool_hook(module, inputs, output):
                captured["pool"] = output if isinstance(output, Tensor) else output[0]

            handles.append(
                self.model.transformer_encoder.layers[
                    pool_layer_idx
                ].register_forward_hook(pool_hook)
            )

        gepc_chunks: list[np.ndarray] = []
        gene_chunks: list[np.ndarray] = []
        pooled_chunks: list[np.ndarray] = []

        autocast_enabled = self.dtype in (torch.float16, torch.bfloat16)
        if autocast_enabled:
            amp_ctx = torch.autocast(device_type=self.device.type, dtype=self.dtype)
        else:
            amp_ctx = contextlib.nullcontext()

        try:
            for start in range(0, n_cells, batch_size):
                end = min(start + batch_size, n_cells)
                src = genes_all[start:end].to(self.device)
                values = values_all[start:end].to(self.device)
                src_key_padding_mask = src.eq(pad_id)
                state["steer_mask"] = self._steering_mask(src_key_padding_mask, mode)

                with amp_ctx:
                    output_dict = self.model(
                        src=src,
                        values=values,
                        src_key_padding_mask=self._forward_padding_mask(
                            src_key_padding_mask
                        ),
                        batch_labels=None,
                        CLS=False,
                        CCE=False,
                        MVC=True,
                        ECS=False,
                    )

                # mvc_output is per-token (B, L); drop the CLS slot at position 0.
                gepc = output_dict["mvc_output"].float()[:, 1:]
                gepc_chunks.append(gepc.cpu().numpy())
                if return_genes:
                    gene_chunks.append(src[:, 1:].cpu().numpy())
                if return_pooled:
                    hidden = _to_dense_hidden(captured["pool"].float())
                    pool_mask = (~src_key_padding_mask).float().unsqueeze(-1)
                    pool_mask[:, 0, :] = 0.0
                    pooled = (hidden * pool_mask).sum(dim=1) / pool_mask.sum(
                        dim=1
                    ).clamp(min=1)
                    pooled_chunks.append(pooled.cpu().numpy())
                    captured.clear()
        finally:
            for handle in handles:
                handle.remove()

        outputs: list[np.ndarray] = [np.concatenate(gepc_chunks, axis=0)]
        if return_genes:
            outputs.append(np.concatenate(gene_chunks, axis=0))
        if return_pooled:
            outputs.append(np.concatenate(pooled_chunks, axis=0))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
