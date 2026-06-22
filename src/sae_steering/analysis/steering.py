# Steering and validity checks for scGPT SAE features.

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

DISEASE_POSITIVE = "lung adenocarcinoma"
DISEASE_HEALTHY = "normal"

EXCLUDE_CELL_TYPES = {"unknown", "nan", ""}

# Spike-test lineage markers (substring match on cell_type)
CELL_TYPE_MARKERS: dict[str, dict[str, list[str]]] = {
    "t cell": {"up": ["CD3D", "CD3E", "CD3G", "TRAC", "CD8A", "IL7R"], "down": []},
    "cd8": {"up": ["CD8A", "CD8B", "CD3D", "GZMK", "NKG7"], "down": []},
    "cd4": {"up": ["CD4", "IL7R", "CD3D", "CD3E", "TRAC"], "down": []},
    "natural killer": {"up": ["NKG7", "GNLY", "KLRD1", "NCAM1", "KLRF1"], "down": []},
    "alveolar macrophage": {"up": ["MARCO", "FABP4", "MCEMP1", "MRC1", "CD68"], "down": []},
    "macrophage": {"up": ["CD68", "LYZ", "CD163", "MSR1", "APOC1"], "down": []},
    "monocyte": {"up": ["LYZ", "S100A8", "S100A9", "FCN1", "VCAN"], "down": []},
    "b cell": {"up": ["CD19", "MS4A1", "CD79A", "CD79B"], "down": []},
    "type 2": {"up": ["SFTPC", "SFTPB", "SFTPA1", "NAPSA", "LAMP3"], "down": []},
    "type 1": {"up": ["AGER", "PDPN", "CAV1"], "down": []},
    "endothelial": {"up": ["PECAM1", "VWF", "CLDN5", "CDH5"], "down": []},
    "fibroblast": {"up": ["COL1A1", "COL1A2", "DCN", "LUM"], "down": []},
}


def resolve_steering_mode(extractor: Any) -> str:
    # Return "gene" or "cls" for extractor's checkpoint config
    no_cls = bool(extractor.args.get("no_cls", False)) if hasattr(extractor, "args") else False
    no_cls = no_cls or bool(getattr(extractor, "_trained_no_cls", False))
    return "gene" if no_cls else "cls"


class CalibratedOffset(NamedTuple):
    # A steering offset plus the scalar it was scaled by

    offset: np.ndarray
    scale: float
    method: str


def _decoder_column(sae: Any, feature_idx: int) -> np.ndarray:
    return sae.decoder.weight[:, feature_idx].detach().cpu().numpy().astype(np.float32)


def feature_activation_std(z_features: np.ndarray, feature_idx: int) -> float:
    # Std of a feature's sparse TopK activations over the scoring cells
    col = np.asarray(z_features)[:, feature_idx]
    return float(np.std(col))


def calibrate_feature_offset(
    sae: Any,
    feature_idx: int,
    z_features: np.ndarray | None,
    *,
    method: Literal["std", "raw"] = "std",
) -> CalibratedOffset:
    # Build the steering offset for an SAE feature
    dec_col = _decoder_column(sae, feature_idx)
    if method == "raw":
        return CalibratedOffset(offset=dec_col, scale=1.0, method="raw")
    if method == "std":
        if z_features is None:
            raise ValueError("method='std' needs z_features to compute the feature scale")
        scale = feature_activation_std(z_features, feature_idx)
        return CalibratedOffset(offset=scale * dec_col, scale=scale, method="std")
    raise ValueError(f"unknown calibration method {method!r}; expected std|raw")


class scGPTSteerer:

    def __init__(
        self,
        extractor: Any,
        sae: Any,
        *,
        layer: int | None = None,
        z_features: np.ndarray | None = None,
        calibration: Literal["std", "raw"] = "std",
    ):
        self.extractor = extractor
        self.sae = sae
        self.layer = extractor.layer if layer is None else layer
        if layer is not None and layer != extractor.layer:
            logger.warning(
                f"scGPTSteerer layer={layer} != extractor.layer={extractor.layer}; "
                "steering still hooks extractor.layer"
            )
        self.z_features = None if z_features is None else np.asarray(z_features)
        self.calibration = calibration
        self.steering_mode = resolve_steering_mode(extractor)
        self._offset_cache: dict[int, CalibratedOffset] = {}
        self._token_cache: dict[int, dict] = {}
        self._adata_keepalive: dict[int, Any] = {}  # hold refs so id() stays unique

    def set_preprocess_reference(self, adata: Any) -> None:
        # Fit mixed-cohort HVG on adata
        self.extractor.fit_preprocess_reference(adata)
        self.unsteer()

    def calibrated_offset(self, feature_idx: int) -> CalibratedOffset:
        # Calibrated offset for feature_idx (cached).
        if feature_idx not in self._offset_cache:
            self._offset_cache[feature_idx] = calibrate_feature_offset(
                self.sae, feature_idx, self.z_features, method=self.calibration
            )
        return self._offset_cache[feature_idx]

    def _tokens(self, adata: Any) -> dict:
        key = id(adata)
        if key not in self._token_cache:
            processed = self.extractor.preprocess(adata)
            self._token_cache[key] = self.extractor.tokenize(processed)
            self._adata_keepalive[key] = adata
        return self._token_cache[key]

    def steer(
        self,
        adata: Any,
        feature_idx: int,
        alpha: float,
        *,
        steering_mode: str = "auto",
        return_genes: bool = False,
        return_pooled: bool = False,
        pool_layer: int | None = None,
    ) -> Any:
        # Forward with alpha * calibrated_offset(feature_idx) added
        cal = self.calibrated_offset(feature_idx)
        tokens = self._tokens(adata)
        return self.extractor.forward_with_steering(
            adata,
            cal.offset,
            alpha,
            return_genes=return_genes,
            return_pooled=return_pooled,
            steering_mode=steering_mode,
            pool_layer=pool_layer,
            tokens=tokens,
        )

    def steer_with_offset(
        self,
        adata: Any,
        offset: np.ndarray,
        alpha: float,
        *,
        steering_mode: str = "auto",
        return_genes: bool = False,
        return_pooled: bool = False,
        pool_layer: int | None = None,
    ) -> Any:
        # Forward with an explicit residual offset, bypassing SAE feature lookup
        tokens = self._tokens(adata)
        return self.extractor.forward_with_steering(
            adata,
            np.asarray(offset, dtype=np.float32),
            alpha,
            return_genes=return_genes,
            return_pooled=return_pooled,
            steering_mode=steering_mode,
            pool_layer=pool_layer,
            tokens=tokens,
        )

    def embed(
        self,
        adata: Any,
        feature_idx: int,
        alpha: float,
        *,
        pool_layer: int | None = None,
        steering_mode: str = "auto",
    ) -> np.ndarray:
        # Steered mean-pooled cell embeddings (n_cells, d_model).
        _, pooled = self.steer(
            adata,
            feature_idx,
            alpha,
            return_pooled=True,
            pool_layer=pool_layer,
            steering_mode=steering_mode,
        )
        return pooled

    def baseline_gepc(self, adata: Any) -> tuple[np.ndarray, np.ndarray]:
        # Unsteered MVC output and per-cell gene tokens (feature-independent).
        tokens = self._tokens(adata)
        zero = np.zeros(self.extractor.args["embsize"], dtype=np.float32)
        return self.extractor.forward_with_steering(
            adata, zero, 0.0, return_genes=True, tokens=tokens
        )

    def unsteer(self) -> None:
        # Drop cached tokens/offsets
        self._token_cache.clear()
        self._adata_keepalive.clear()
        self._offset_cache.clear()

    def __enter__(self) -> scGPTSteerer:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.unsteer()
        return False


def compute_expression_signature(
    steerer: scGPTSteerer,
    adata: Any,
    feature_idx: int,
    alpha: float,
    *,
    baseline: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    # Per-gene GEPC *shift* (steer-on minus baseline) aggregated by gene token
    from sae_steering.analysis import feature_discovery as fd

    if baseline is None:
        baseline = steerer.baseline_gepc(adata)
    mvc_base, genes = baseline
    mvc_steer, _ = steerer.steer(adata, feature_idx, alpha, return_genes=True)
    return fd._aggregate_gene_shift(mvc_steer - mvc_base, genes, steerer.extractor)


def scatter_signature_to_vocab(signature_df: pd.DataFrame, vocab_size: int) -> np.ndarray:
    # Scatter an aggregated signature into a dense (vocab_size,) vector
    vec = np.zeros(int(vocab_size), dtype=np.float32)
    tokens = signature_df["gene_token"].to_numpy().astype(int)
    vec[tokens] = signature_df["expected_shift"].to_numpy(dtype=np.float32)
    return vec


def _maybe_subset(adata: Any, mask: np.ndarray) -> Any:
    mask = np.asarray(mask, dtype=bool)
    if mask.all():
        return adata
    return adata[mask]


def _subsample(adata: Any, n_cells: int, seed: int) -> Any:
    if adata.n_obs <= n_cells:
        return adata
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(adata.n_obs, size=n_cells, replace=False))
    return adata[idx]


def prepare_steering_cohort(
    steerer: scGPTSteerer,
    full_adata: Any,
    *,
    steer_on: Literal["healthy", "all"] = "healthy",
    n_cells: int,
    seed: int = 0,
) -> Any:
    # Fit preprocess on the full cohort, then return a steering cell subset.
    steerer.set_preprocess_reference(full_adata)
    adata = full_adata
    if steer_on == "healthy":
        adata = adata[adata.obs["disease"].astype(str) == DISEASE_HEALTHY].copy()
    return _subsample(adata, n_cells, seed)


def select_validity_cells(
    healthy_adata: Any,
    luad_adata: Any,
    *,
    n_cells: int = 200,
    seed: int = 0,
    assay_match: str | None = "10x 3' v2",
    min_after_assay: int = 20,
) -> tuple[Any, Any, bool]:
    healthy = _maybe_subset(
        healthy_adata,
        (healthy_adata.obs["disease"].astype(str) == DISEASE_HEALTHY).to_numpy(),
    )
    luad = _maybe_subset(
        luad_adata,
        (luad_adata.obs["disease"].astype(str) == DISEASE_POSITIVE).to_numpy(),
    )

    assay_applied = False
    if assay_match is not None and "assay" in healthy.obs.columns:
        h_mask = (healthy.obs["assay"].astype(str) == assay_match).to_numpy()
        l_mask = (luad.obs["assay"].astype(str) == assay_match).to_numpy()
        if h_mask.sum() >= min_after_assay and l_mask.sum() >= min_after_assay:
            healthy = _maybe_subset(healthy, h_mask)
            luad = _maybe_subset(luad, l_mask)
            assay_applied = True
        else:
            logger.warning(
                f"assay_match={assay_match!r} leaves "
                f"{int(h_mask.sum())} healthy / {int(l_mask.sum())} LUAD "
                f"(< {min_after_assay}); skipping assay filter"
            )

    healthy = _subsample(healthy, n_cells, seed)
    luad = _subsample(luad, n_cells, seed)
    return healthy, luad, assay_applied


def _cosine_rows(emb: np.ndarray, vec: np.ndarray) -> np.ndarray:
    # Cosine similarity of each row of emb with vec.
    en = np.linalg.norm(emb, axis=1)
    vn = float(np.linalg.norm(vec))
    denom = np.clip(en * vn, 1e-8, None)
    return (emb @ vec) / denom


def _monotonic_spearman(xs: list[float], ys: list[float]) -> float:
    # Spearman correlation of ys against xs
    if len(set(np.round(ys, 12))) < 2:
        return float("nan")
    rho = spearmanr(xs, ys).correlation
    return float(rho) if rho is not None else float("nan")


def _phase4_sanity(
    steerer: scGPTSteerer,
    baseline_adata: Any,
    feature_idx: int,
    *,
    alpha: float = 1.0,
    spearman_min: float = 0.3,
) -> bool:
    from sae_steering.analysis import feature_discovery as fd

    base = steerer.baseline_gepc(baseline_adata)
    decode_df = fd.decode_feature_to_gene_weights(
        steerer.extractor, steerer.sae, feature_idx, baseline_adata, alpha=alpha, baseline=base
    )
    res = fd.steering_sanity_check(
        steerer.extractor,
        steerer.sae,
        feature_idx,
        baseline_adata,
        decode_df,
        alpha=alpha,
        baseline=base,
        spearman_min=spearman_min,
    )
    return bool(res["passes"])


def validity_check_steering(
    steerer: scGPTSteerer,
    healthy_adata: Any,
    luad_adata: Any,
    feature_idx: int,
    alpha_grid: tuple[float, ...] = (-3, -1, 0, 1, 3),
    *,
    n_cells: int = 200,
    seed: int = 0,
    embedding_layer: Literal["sae", "last"] = "sae",
    metric: Literal["centroid_projection", "cosine_to_luad", "distance_ratio"] = "centroid_projection",
    assay_match: str | None = "10x 3' v2",
    monotonicity_spearman_min: float = 0.8,
    steering_mode: str = "auto",
    require_phase4_sanity: bool = False,
) -> dict[str, Any]:
    extractor = steerer.extractor
    applied_mode = extractor._resolve_steering_mode(steering_mode)
    if embedding_layer == "sae":
        pool_layer = None
    elif embedding_layer == "last":
        pool_layer = int(extractor.args["nlayers"]) - 1
    else:
        raise ValueError(f"unknown embedding_layer {embedding_layer!r}; expected sae|last")

    healthy, luad, assay_applied = select_validity_cells(
        healthy_adata, luad_adata, n_cells=n_cells, seed=seed, assay_match=assay_match
    )

    cal = steerer.calibrated_offset(feature_idx)
    feature_std = float(cal.scale)

    # Unsteered embeddings -> centroids and healthy-axis origin.
    emb_healthy0 = steerer.embed(healthy, feature_idx, 0.0, pool_layer=pool_layer, steering_mode=steering_mode)
    emb_luad0 = steerer.embed(luad, feature_idx, 0.0, pool_layer=pool_layer, steering_mode=steering_mode)
    mu_h = emb_healthy0.mean(axis=0)
    mu_l = emb_luad0.mean(axis=0)
    direction = mu_l - mu_h
    dnorm = float(np.linalg.norm(direction))
    unit = direction / dnorm if dnorm > 0 else direction

    alphas = sorted(float(a) for a in alpha_grid)
    proj_by_alpha: dict[float, float] = {}
    cos_by_alpha: dict[float, float] = {}
    dist_by_alpha: dict[float, float] = {}
    for a in alphas:
        emb = (
            emb_healthy0
            if a == 0.0
            else steerer.embed(healthy, feature_idx, a, pool_layer=pool_layer, steering_mode=steering_mode)
        )
        proj_by_alpha[a] = float(((emb - mu_h) @ unit).mean())
        cos_by_alpha[a] = float(_cosine_rows(emb, mu_l).mean())
        ratio = np.linalg.norm(emb - mu_l, axis=1) / np.clip(
            np.linalg.norm(emb - mu_h, axis=1), 1e-8, None
        )
        dist_by_alpha[a] = float(ratio.mean())

    gate_series = {
        "centroid_projection": proj_by_alpha,
        "cosine_to_luad": cos_by_alpha,
        "distance_ratio": dist_by_alpha,
    }[metric]
    mono = _monotonic_spearman(alphas, [gate_series[a] for a in alphas])
    # distance_ratio shrinks toward LUAD; the other two grow
    if metric == "distance_ratio":
        passes = bool(np.isfinite(mono) and mono <= -monotonicity_spearman_min)
    else:
        passes = bool(np.isfinite(mono) and mono >= monotonicity_spearman_min)

    passes_sanity: bool | None = None
    if require_phase4_sanity:
        passes_sanity = _phase4_sanity(steerer, healthy, feature_idx)
        passes = passes and bool(passes_sanity)

    return {
        "feature_idx": int(feature_idx),
        "feature_std": feature_std,
        "calibration": steerer.calibration,
        "steering_mode": applied_mode,
        "embedding_layer": embedding_layer,
        "metric": metric,
        "assay_match": assay_match if assay_applied else None,
        "assay_applied": bool(assay_applied),
        "n_healthy": int(emb_healthy0.shape[0]),
        "n_luad": int(emb_luad0.shape[0]),
        "alpha_grid": alphas,
        "luad_axis_norm": dnorm,
        "centroid_projection_by_alpha": {str(a): proj_by_alpha[a] for a in alphas},
        "cosine_to_luad_by_alpha": {str(a): cos_by_alpha[a] for a in alphas},
        "distance_ratio_by_alpha": {str(a): dist_by_alpha[a] for a in alphas},
        "monotonicity_spearman": float(mono) if np.isfinite(mono) else None,
        "passes_validity": passes,
        "passes_sanity_optional": passes_sanity,
    }


def plot_steering_umap(
    steerer: scGPTSteerer,
    healthy_adata: Any,
    luad_adata: Any,
    feature_idx: int,
    alpha_grid: tuple[float, ...] = (-3, -1, 0, 1, 3),
    *,
    n_cells: int = 200,
    seed: int = 0,
    embedding_layer: Literal["sae", "last"] = "sae",
    assay_match: str | None = "10x 3' v2",
    steering_mode: str = "auto",
    out_path: str | Path = "data/steering/umap_shift_panel.png",
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        logger.warning(f"umap/matplotlib unavailable ({exc}); skipping UMAP plot")
        return None

    pool_layer = None if embedding_layer == "sae" else int(steerer.extractor.args["nlayers"]) - 1
    healthy, luad, _ = select_validity_cells(
        healthy_adata, luad_adata, n_cells=n_cells, seed=seed, assay_match=assay_match
    )
    emb_h = steerer.embed(healthy, feature_idx, 0.0, pool_layer=pool_layer, steering_mode=steering_mode)
    emb_l = steerer.embed(luad, feature_idx, 0.0, pool_layer=pool_layer, steering_mode=steering_mode)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed)
    reducer.fit(np.vstack([emb_h, emb_l]))
    h2d = reducer.transform(emb_h)
    l2d = reducer.transform(emb_l)

    cents = {0.0: reducer.transform(emb_h.mean(axis=0, keepdims=True))[0]}
    for a in sorted(float(x) for x in alpha_grid):
        if a == 0.0:
            continue
        emb_a = steerer.embed(healthy, feature_idx, a, pool_layer=pool_layer, steering_mode=steering_mode)
        cents[a] = reducer.transform(emb_a.mean(axis=0, keepdims=True))[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(l2d[:, 0], l2d[:, 1], s=8, c="tab:red", alpha=0.4, label="LUAD (unsteered)")
    ax.scatter(h2d[:, 0], h2d[:, 1], s=8, c="tab:blue", alpha=0.4, label="healthy (unsteered)")
    order = sorted(cents)
    ax.plot(
        [cents[a][0] for a in order],
        [cents[a][1] for a in order],
        "-o",
        c="black",
        label="healthy centroid vs alpha",
    )
    for a in order:
        ax.annotate(f"α={a:g}", cents[a], fontsize=8)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"feature {feature_idx} steering shift ({embedding_layer} embedding)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote UMAP shift panel to {out_path}")
    return out_path


def markers_in_hvg_fraction(adata: Any, markers: list[str]) -> float:
    # Fraction of markers present in adata's gene set
    markers = [str(m) for m in markers]
    if not markers:
        return float("nan")
    if "gene_name" in adata.var.columns:
        present = set(adata.var["gene_name"].astype(str))
    else:
        present = set(map(str, adata.var_names))
    hits = sum(m in present for m in markers)
    return hits / len(markers)


def markers_for_cell_type(cell_type: str) -> dict[str, list[str]] | None:
    # Look up marker panel for a CellxGene cell-type label (substring match).
    name = str(cell_type).lower()
    for key, panel in CELL_TYPE_MARKERS.items():
        if key in name:
            return panel
    return None


def _pick_spike_cell_types(obs: pd.DataFrame, cell_types: tuple[str, str] | None) -> tuple[str, str]:
    if cell_types is not None:
        return str(cell_types[0]), str(cell_types[1])
    vc = obs["cell_type"].astype(str).value_counts()
    vc = vc[~vc.index.str.lower().isin(EXCLUDE_CELL_TYPES)]
    if len(vc) < 2:
        raise ValueError("need >= 2 non-'unknown' cell types in obs for the spike test")
    return str(vc.index[0]), str(vc.index[1])


def spike_test_cell_type_steering(
    extractor: Any,
    sae: Any,
    adata: Any,
    activations: np.ndarray,
    obs: pd.DataFrame,
    *,
    cell_types: tuple[str, str] | None = None,
    alpha: float = 1.0,
    n_steer_cells: int = 200,
    seed: int = 0,
    min_firing_rate: float = 0.001,
    min_markers_hvg: float = 0.3,
    n_top_genes: int = 50,
) -> dict[str, Any]:
    from sae_steering.analysis import feature_discovery as fd

    obs = obs.reset_index(drop=True)
    source, other = _pick_spike_cell_types(obs, cell_types)
    logger.info(f"spike test: source={source!r} vs other={other!r}")

    # 1-2
    z, _ = fd.encode_features(sae, activations)
    ct = obs["cell_type"].astype(str).to_numpy()
    m_src = ct == source
    m_oth = ct == other
    if m_src.sum() < 2 or m_oth.sum() < 2:
        raise ValueError(
            f"too few cells: {int(m_src.sum())} {source!r} / {int(m_oth.sum())} {other!r}"
        )
    firing = (z != 0).mean(axis=0)
    live = np.where(firing >= min_firing_rate)[0]
    ds = np.array([fd.cohens_d(z[m_src, f], z[m_oth, f]) for f in live])
    top_feature = int(live[int(np.nanargmax(ds))])
    top_d = float(np.nanmax(ds))
    logger.info(f"spike test: top feature {top_feature} (cohens_d source>other = {top_d:.2f})")

    # 3. Steer the source lineage's raw cells.
    src_cells = _subsample(
        _maybe_subset(adata, (adata.obs["cell_type"].astype(str) == source).to_numpy()),
        n_steer_cells,
        seed,
    )

    # Preflight: are the source markers even in the HVG set?
    panel = markers_for_cell_type(source)
    up_markers = panel["up"] if panel else []
    extractor.fit_preprocess_reference(adata)
    processed = extractor.preprocess(src_cells.copy())
    marker_frac = markers_in_hvg_fraction(processed, up_markers) if up_markers else float("nan")
    warning = None
    if not up_markers:
        warning = f"no marker panel for source cell type {source!r}; signature reported without check"
    elif not np.isfinite(marker_frac) or marker_frac < min_markers_hvg:
        warning = (
            f"only {marker_frac:.2f} of {source!r} up-markers in HVG "
            f"(< {min_markers_hvg}); GEPC readout may be silent"
        )
    if warning:
        logger.warning(f"spike test: {warning}")

    steerer = scGPTSteerer(extractor, sae, z_features=z, calibration="std")
    steerer.set_preprocess_reference(adata)
    sig = compute_expression_signature(steerer, src_cells, top_feature, alpha)
    sig_sorted = sig.sort_values("expected_shift", ascending=False).reset_index(drop=True)
    rank_of = {g: i for i, g in enumerate(sig_sorted["gene_symbol"])}
    n_genes = len(sig_sorted)
    marker_ranks = {g: rank_of.get(g) for g in up_markers}
    present_ranks = [r for r in marker_ranks.values() if r is not None]
    # "moved up" if the markers land in the top tertile of the signature on average.
    median_rank = float(np.median(present_ranks)) if present_ranks else float("nan")
    markers_moved_up = bool(present_ranks) and median_rank < (n_genes / 3.0)

    return {
        "source_cell_type": source,
        "other_cell_type": other,
        "n_source_cells": int(m_src.sum()),
        "n_other_cells": int(m_oth.sum()),
        "top_feature": top_feature,
        "cohens_d_source_vs_other": top_d,
        "alpha": float(alpha),
        "up_markers": up_markers,
        "markers_in_hvg_fraction": None if np.isnan(marker_frac) else marker_frac,
        "marker_ranks": marker_ranks,
        "n_signature_genes": int(n_genes),
        "median_marker_rank": None if np.isnan(median_rank) else median_rank,
        "markers_moved_up": markers_moved_up,
        "top_signature_genes": sig_sorted["gene_symbol"].head(n_top_genes).tolist(),
        "warning": warning,
    }


def geneformer_rank_delta_signature(
    geneformer_steerer: Any, cells: Any, feature_idx: int, alpha: float
) -> np.ndarray:
    # Rank-delta readout for cross-model validation
    raise NotImplementedError(
        "geneformer rank-delta steering is deferred to Phase 5b "
        "(no masked-LM logits steering path in geneformer_wrapper yet)"
    )
