# Spearman correlation between steering shifts and Tahoe drug DE.


from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr, spearmanr

from sae_steering.data.gene_mapping import get_or_build_from_gene_info


class InsufficientOverlapError(ValueError):
    pass


def _normalize_gene_id(gene_id: object) -> str | None:
    if gene_id is None or pd.isna(gene_id):
        return None
    value = str(gene_id)
    if not value or value.lower() == "nan":
        return None
    return value.split(".")[0]


def _gene_mapping(gene_info_path: Path) -> pd.DataFrame:
    gene_info_path = Path(gene_info_path)
    return get_or_build_from_gene_info(gene_info_path, gene_info_path.parent / "gene_mapping.parquet")


def _unique_symbol_to_gene_id(mapping: pd.DataFrame) -> dict[str, str]:
    clean = mapping.dropna(subset=["symbol", "ensembl_id"]).copy()
    clean["gene_id"] = clean["ensembl_id"].map(_normalize_gene_id)
    counts = clean.groupby("symbol")["gene_id"].nunique()
    unique_symbols = counts[counts == 1].index
    return (
        clean[clean["symbol"].isin(unique_symbols)]
        .drop_duplicates("symbol")
        .set_index("symbol")["gene_id"]
        .to_dict()
    )


def _standardize_signature(
    df: pd.DataFrame,
    *,
    score_col: str,
    out_score_col: str,
    gene_info: pd.DataFrame,
) -> pd.DataFrame:
    if score_col not in df.columns:
        raise KeyError(f"signature missing score column {score_col!r}")

    out = pd.DataFrame({out_score_col: pd.to_numeric(df[score_col], errors="coerce")})
    out["gene_id"] = df["gene_id"].map(_normalize_gene_id) if "gene_id" in df.columns else None

    if "gene_symbol" in df.columns:
        out["gene_symbol"] = df["gene_symbol"].astype(str)
    elif "gene_name" in df.columns:
        out["gene_symbol"] = df["gene_name"].astype(str)
    else:
        out["gene_symbol"] = None

    unresolved = out["gene_id"].isna() & out["gene_symbol"].notna()
    if unresolved.any():
        out.loc[unresolved, "gene_id"] = out.loc[unresolved, "gene_symbol"].map(
            _unique_symbol_to_gene_id(gene_info)
        )
    return out.dropna(subset=[out_score_col]).copy()


def _collapse_by_gene(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    keyed = df.dropna(subset=["gene_id"]).copy()
    if keyed.empty:
        return keyed
    return keyed.groupby("gene_id", as_index=False).agg(
        gene_symbol=("gene_symbol", "first"), **{score_col: (score_col, "mean")}
    )


def align_gene_universes(
    steer_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    *,
    steer_score_col: str = "expected_shift",
    drug_score_col: str = "log2_fold_change",
    gene_info_path: Path,
    hk_genes: list[str] | None = None,
    drop_zero_variance: bool = True,
    min_genes: int = 200,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    # Return aligned steering and drug score arrays plus per-gene metadata
    gene_info = _gene_mapping(Path(gene_info_path))
    steer = _standardize_signature(
        steer_df, score_col=steer_score_col, out_score_col="steer_score", gene_info=gene_info
    )
    drug = _standardize_signature(
        drug_df, score_col=drug_score_col, out_score_col="drug_score", gene_info=gene_info
    )
    stats = {
        "steer_input": int(len(steer_df)),
        "drug_input": int(len(drug_df)),
        "steer_with_gene_id": int(steer["gene_id"].notna().sum()),
        "drug_with_gene_id": int(drug["gene_id"].notna().sum()),
    }

    steer = _collapse_by_gene(steer, "steer_score")
    drug = _collapse_by_gene(drug, "drug_score")
    meta = steer.merge(drug, on="gene_id", how="inner", suffixes=("_steer", "_drug"))
    if "gene_symbol_steer" in meta.columns:
        meta["gene_symbol"] = meta["gene_symbol_steer"].fillna(meta["gene_symbol_drug"])
    meta = meta[["gene_id", "gene_symbol", "steer_score", "drug_score"]].dropna(
        subset=["steer_score", "drug_score"]
    )

    if hk_genes:
        hk = {str(g) for g in hk_genes}
        meta = meta[
            ~meta["gene_id"].astype(str).isin(hk) & ~meta["gene_symbol"].astype(str).isin(hk)
        ].copy()

    if drop_zero_variance and len(meta) > 0:
        steer_std = float(np.nanstd(meta["steer_score"].to_numpy(dtype=float)))
        drug_std = float(np.nanstd(meta["drug_score"].to_numpy(dtype=float)))
        if steer_std == 0.0 or drug_std == 0.0:
            raise InsufficientOverlapError(
                "aligned scores have zero variance "
                f"(n={len(meta)}, steer_std={steer_std:.3g}, drug_std={drug_std:.3g})"
            )

    if len(meta) < min_genes:
        raise InsufficientOverlapError(
            f"only {len(meta)} aligned genes; need >= {min_genes}; breakdown={stats}"
        )

    meta = meta.sort_values("gene_id").reset_index(drop=True)
    return (
        meta["steer_score"].to_numpy(dtype=float),
        meta["drug_score"].to_numpy(dtype=float),
        meta,
    )


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    res = spearmanr(x, y)
    rho = None if res.correlation is None or np.isnan(res.correlation) else float(res.correlation)
    pval = None if res.pvalue is None or np.isnan(res.pvalue) else float(res.pvalue)
    return rho, pval


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    r, p = pearsonr(x, y)
    return float(r), float(p)


def _top_contributors(meta_df: pd.DataFrame, n: int = 20) -> list[dict[str, Any]]:
    x = meta_df["steer_score"].to_numpy(dtype=float)
    y = meta_df["drug_score"].to_numpy(dtype=float)
    sx = (x - x.mean()) / np.clip(x.std(ddof=0), 1e-12, None)
    sy = (y - y.mean()) / np.clip(y.std(ddof=0), 1e-12, None)
    tmp = meta_df.copy()
    tmp["signed_contrib"] = sx * sy / max(len(meta_df) - 1, 1)
    tmp = tmp.reindex(tmp["signed_contrib"].abs().sort_values(ascending=False).index).head(n)
    return [
        {
            "gene_id": str(row.gene_id),
            "gene_symbol": None if pd.isna(row.gene_symbol) else str(row.gene_symbol),
            "steer_score": float(row.steer_score),
            "drug_score": float(row.drug_score),
            "signed_contrib": float(row.signed_contrib),
        }
        for row in tmp.itertuples(index=False)
    ]


def _rges(steer: np.ndarray, drug: np.ndarray) -> float:
    n = len(steer)
    if n == 0:
        return float("nan")
    drug_rank = pd.Series(drug).rank(method="average", ascending=False).to_numpy()
    top_n = max(1, int(np.ceil(0.1 * n)))
    up = np.argsort(steer)[-top_n:]
    down = np.argsort(steer)[:top_n]
    up_es = 1.0 - (drug_rank[up].mean() - 1.0) / max(n - 1, 1)
    down_es = (drug_rank[down].mean() - 1.0) / max(n - 1, 1)
    return float(up_es - down_es)


def compute_sign_agreement(
    steer_aligned: np.ndarray,
    drug_aligned: np.ndarray,
    meta_df: pd.DataFrame,
    *,
    method: Literal["spearman", "pearson", "both"] = "spearman",
    compute_rges: bool = False,
) -> dict[str, Any]:
    # Compute rank/linear agreement and genes driving the linear correlation.
    steer = np.asarray(steer_aligned, dtype=float)
    drug = np.asarray(drug_aligned, dtype=float)
    if steer.shape != drug.shape:
        raise ValueError(f"aligned arrays have different shapes: {steer.shape} vs {drug.shape}")
    if len(steer) != len(meta_df):
        raise ValueError("meta_df length must match aligned arrays")
    if len(steer) < 2:
        raise InsufficientOverlapError("need at least two aligned genes")

    spearman_rho = spearman_pvalue = pearson_r = pearson_pvalue = None
    if method in {"spearman", "both"}:
        spearman_rho, spearman_pvalue = _safe_spearman(steer, drug)
    if method in {"pearson", "both"}:
        pearson_r, pearson_pvalue = _safe_pearson(steer, drug)

    return {
        "spearman_rho": spearman_rho,
        "spearman_pvalue": spearman_pvalue,
        "pearson_r": pearson_r,
        "pearson_pvalue": pearson_pvalue,
        "rges": _rges(steer, drug) if compute_rges else None,
        "n_genes": int(len(steer)),
        "top_contributors": _top_contributors(meta_df),
    }


def _signature_from_offset(
    steerer: Any,
    adata: Any,
    offset: np.ndarray,
    alpha: float,
    *,
    baseline: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    from sae_steering.analysis import feature_discovery as fd

    if baseline is None:
        baseline = steerer.baseline_gepc(adata)
    mvc_base, genes = baseline
    mvc_steer, _ = steerer.steer_with_offset(adata, offset, alpha, return_genes=True)
    return fd._aggregate_gene_shift(mvc_steer - mvc_base, genes, steerer.extractor)


def permutation_null(
    steerer: Any,
    adata: Any,
    drug_df: pd.DataFrame,
    *,
    n_permutations: int = 100,
    alpha: float = 1.0,
    seed: int = 0,
    gene_info_path: Path,
    hk_genes: list[str] | None = None,
    min_genes: int = 200,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    decoder = getattr(getattr(steerer, "sae", None), "decoder", None)
    if decoder is None:
        raise AttributeError("steerer.sae.decoder is required for latent permutation null")
    d_model = int(decoder.weight.shape[0])
    baseline = steerer.baseline_gepc(adata)
    null = np.full(int(n_permutations), np.nan, dtype=float)

    for i in range(int(n_permutations)):
        direction = rng.normal(size=d_model).astype(np.float32)
        direction /= np.clip(np.linalg.norm(direction), 1e-12, None)
        steer_df = _signature_from_offset(steerer, adata, direction, alpha, baseline=baseline)
        try:
            steer_arr, drug_arr, meta = align_gene_universes(
                steer_df,
                drug_df,
                gene_info_path=gene_info_path,
                hk_genes=hk_genes,
                min_genes=min_genes,
            )
            null[i] = compute_sign_agreement(steer_arr, drug_arr, meta)["spearman_rho"]
        except InsufficientOverlapError as exc:
            logger.warning(f"permutation {i} skipped: {exc}")
        if (i + 1) % 10 == 0:
            logger.info(f"permutation null {i + 1}/{n_permutations}")
    return null


def empirical_pvalue(
    observed: float | None,
    null: np.ndarray,
    *,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
) -> float | None:
    # Empirical p-value with add-one smoothing.
    if observed is None or np.isnan(observed):
        return None
    clean = np.asarray(null, dtype=float)
    clean = clean[~np.isnan(clean)]
    if len(clean) == 0:
        return None
    if alternative == "greater":
        count = int(np.sum(clean >= observed))
    elif alternative == "less":
        count = int(np.sum(clean <= observed))
    elif alternative == "two-sided":
        count = int(np.sum(np.abs(clean) >= abs(observed)))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return float((1 + count) / (1 + len(clean)))


def bh_fdr(pvalues: list[float | None]) -> list[float | None]:
    # Benjamini-Hochberg adjusted p-values, preserving None/NaN positions.
    p = np.array([np.nan if v is None else float(v) for v in pvalues], dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    if len(valid) == 0:
        return [None] * len(pvalues)
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.clip(ranked, 0.0, 1.0)
    return [None if np.isnan(v) else float(v) for v in out]


def bonferroni(pvalues: list[float | None]) -> list[float | None]:
    p = [None if v is None or np.isnan(float(v)) else float(v) for v in pvalues]
    n = sum(v is not None for v in p)
    return [None if v is None else min(1.0, v * n) for v in p]


def _parse_signature_meta(path: Path) -> dict[str, Any]:
    meta_path = Path(str(path).replace("_scgpt.parquet", "_meta.json"))
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    stem = path.name.replace("_scgpt.parquet", "")
    parts = stem.split("__")
    meta = {
        "cell_line": parts[0] if len(parts) > 0 else None,
        "drug": parts[1] if len(parts) > 1 else None,
    }
    if len(parts) > 2:
        match = re.fullmatch(r"([0-9.eE+-]+)(.*)", parts[2])
        if match:
            meta["concentration"] = float(match.group(1))
            meta["concentration_unit"] = match.group(2)
    return meta


def full_hypothesis_test(
    steerer: Any,
    adata: Any,
    validated_features: list[int],
    drug_signature_paths: list[Path],
    *,
    alpha_grid: tuple[float, ...] = (1.0,),
    n_permutations: int = 100,
    comparison_context: str = "cross_context",
    gene_info_path: Path,
    run_mode: str = "pilot",
    hk_genes: list[str] | None = None,
    min_genes: int = 200,
    compute_rges: bool = False,
    fdr_threshold: float = 0.05,
    permutation_alternative: Literal["greater", "less", "two-sided"] = "greater",
) -> pd.DataFrame:
    from sae_steering.analysis import steering as st

    rows: list[dict[str, Any]] = []
    baseline = steerer.baseline_gepc(adata)
    for feature_idx in validated_features:
        for alpha in alpha_grid:
            steer_df = st.compute_expression_signature(
                steerer, adata, int(feature_idx), float(alpha), baseline=baseline
            )
            for drug_path in drug_signature_paths:
                drug_path = Path(drug_path)
                drug_df = pd.read_parquet(drug_path)
                sig_meta = _parse_signature_meta(drug_path)
                row = {
                    "feature_idx": int(feature_idx),
                    "cell_line": sig_meta.get("cell_line"),
                    "drug": sig_meta.get("drug"),
                    "concentration": sig_meta.get("concentration"),
                    "concentration_unit": sig_meta.get("concentration_unit"),
                    "alpha": float(alpha),
                    "model": "scgpt",
                    "comparison_context": comparison_context,
                    "run_mode": run_mode,
                    "signature_path": str(drug_path),
                    "error": None,
                }
                try:
                    steer_arr, drug_arr, aligned = align_gene_universes(
                        steer_df,
                        drug_df,
                        gene_info_path=gene_info_path,
                        hk_genes=hk_genes,
                        min_genes=min_genes,
                    )
                    metrics = compute_sign_agreement(
                        steer_arr, drug_arr, aligned, method="both", compute_rges=compute_rges
                    )
                    null = permutation_null(
                        steerer,
                        adata,
                        drug_df,
                        n_permutations=n_permutations,
                        alpha=float(alpha),
                        seed=int(feature_idx) * 1009 + int(abs(alpha) * 100),
                        gene_info_path=gene_info_path,
                        hk_genes=hk_genes,
                        min_genes=min_genes,
                    )
                    row.update(metrics)
                    row["perm_pvalue"] = empirical_pvalue(
                        metrics["spearman_rho"], null, alternative=permutation_alternative
                    )
                    row["null_mean"] = float(np.nanmean(null)) if np.isfinite(null).any() else None
                    row["null_std"] = float(np.nanstd(null)) if np.isfinite(null).any() else None
                    row["top_contributors_json"] = json.dumps(metrics["top_contributors"])
                    row.pop("top_contributors", None)
                except Exception as exc:  # noqa: BLE001 - keep grid rows inspectable
                    logger.warning(f"hypothesis row failed ({feature_idx}, {drug_path.name}): {exc}")
                    row.update(
                        {
                            "spearman_rho": None,
                            "spearman_pvalue": None,
                            "pearson_r": None,
                            "pearson_pvalue": None,
                            "rges": None,
                            "n_genes": 0,
                            "perm_pvalue": None,
                            "null_mean": None,
                            "null_std": None,
                            "top_contributors_json": "[]",
                            "error": str(exc),
                        }
                    )
                rows.append(row)

    results = pd.DataFrame(rows)
    p_for_fdr = results["perm_pvalue"].where(results["perm_pvalue"].notna(), results["spearman_pvalue"])
    p_list = [None if pd.isna(v) else float(v) for v in p_for_fdr]
    results["padj_bh"] = bh_fdr(p_list)
    results["padj_bonferroni"] = bonferroni(p_list)
    results["passes_fdr"] = results["padj_bh"].map(
        lambda v: bool(v < fdr_threshold) if v is not None and not pd.isna(v) else False
    )
    return results


def cross_validation_controls(
    steerer: Any,
    adata: Any,
    validated_features: list[int],
    pair_manifest: pd.DataFrame,
    *,
    negative_moa_pairs: list[dict] | None = None,
    alpha: float = 1.0,
) -> pd.DataFrame:
    # Select Tahoe-discovered negative control candidates from the manifest.
    del steerer, adata, negative_moa_pairs, alpha
    rows: list[dict[str, Any]] = []
    if pair_manifest.empty or not validated_features:
        return pd.DataFrame(rows)
    manifest = pair_manifest.copy()
    for feature_idx in validated_features:
        first = manifest.iloc[0]
        moa = manifest.get("moa_broad", pd.Series("", index=manifest.index)).astype(str)
        different_moa = manifest[moa != str(first.get("moa_broad", ""))]
        control = different_moa.iloc[0] if not different_moa.empty else first
        rows.append(
            {
                "control_type": "drug_specificity",
                "feature_idx": int(feature_idx),
                "cell_line": control.get("cell_line"),
                "drug": control.get("drug"),
                "spearman_rho": None,
                "perm_pvalue": None,
                "interpretation": "manifest-selected candidate; run via full_hypothesis_test for metrics",
            }
        )
    return pd.DataFrame(rows)


def cross_model_test(*args: Any, **kwargs: Any) -> pd.DataFrame:
    # Phase 7b hook for Geneformer/scGPT concordance.
    del args, kwargs
    raise NotImplementedError(
        "Phase 7b: requires Geneformer steering (Phase 5b) and feature matching spec"
    )


def spearman_against_drug_de(steering_shift, drug_de_table):
    # Legacy Series wrapper
    if isinstance(steering_shift, pd.Series) and isinstance(drug_de_table, pd.Series):
        joined = pd.concat([steering_shift, drug_de_table], axis=1, join="inner").dropna()
        joined.columns = ["steer_score", "drug_score"]
        rho, pvalue = _safe_spearman(
            joined["steer_score"].to_numpy(dtype=float),
            joined["drug_score"].to_numpy(dtype=float),
        )
        return {"spearman_rho": rho, "spearman_pvalue": pvalue, "n_genes": int(len(joined))}
    raise NotImplementedError(
        "Use align_gene_universes(...) + compute_sign_agreement(...) for DataFrame signatures"
    )
