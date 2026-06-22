# Score and annotate SAE features for disease separation.

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.stats import hypergeom, mannwhitneyu, spearmanr

DISEASE_POSITIVE = "lung adenocarcinoma"

GENE_SET_LIBRARIES = [
    "MSigDB_Hallmark_2020",
    "KEGG_2021_Human",
    "Reactome_2022",
    "GO_Biological_Process_2023",
]

# Cell-cycle S phase markers
S_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM7", "MCM4", "RRM1", "UNG", "GINS2",
    "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "CENPU", "HELLS", "RFC2",
    "POLR1B", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7",
    "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN",
    "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "MRPL36",
    "E2F8",
]
G2M_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80",
    "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "PIMREG",
    "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANLN",
    "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "JPT1", "CDC20", "TTK",
    "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2",
    "KIF23", "HMMR", "AURKA", "PSRC1", "ANP32E", "G2E3", "GAS2L3", "CBX5",
    "CENPA",
]

# Default verdict gate thresholds (mirror configs/discover_disease_features.yaml).
DEFAULT_GATES: dict[str, float] = {
    "depth_corr": 0.5,
    "assay_eta_sq": 0.3,
    "cohens_d_donor_min": 0.5,
    "cohens_d_platform_min": 0.3,
    "wilcoxon_padj_max": 0.05,
    "min_firing_rate": 0.001,
}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    # Cohen's d with pooled standard deviation
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def _ranksum_p(a: np.ndarray, b: np.ndarray) -> float:
    # Two-sided Wilcoxon rank-sum (Mann-Whitney) p-value
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 1 or len(b) < 1:
        return float("nan")
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    # Pearson r, returning 0.0 when either vector is constant or has NaNs.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return 0.0
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    # Single-feature ranking AUROC (no fitting)
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores, dtype=float)))


def _anova_eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    # One-way ANOVA eta^2: fraction of activation variance explained by group.
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    grand = values.mean()
    ss_total = float(((values - grand) ** 2).sum())
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        v = values[groups == g]
        ss_between += len(v) * (v.mean() - grand) ** 2
    return float(ss_between / ss_total)


def _bh_correct(pvals: pd.Series) -> np.ndarray:
    # Benjamini-Hochberg FDR over the finite p-values
    from statsmodels.stats.multitest import multipletests

    p = pvals.to_numpy(dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.sum() > 0:
        out[finite] = multipletests(p[finite], method="fdr_bh")[1]
    return out


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@torch.no_grad()
def encode_features(
    sae: Any, activations: np.ndarray, batch_size: int = 8192
) -> tuple[np.ndarray, np.ndarray]:
    # Return (sparse TopK activations z, dense pre-activations z_pre) as numpy
    sae.eval()
    x_all = torch.as_tensor(np.asarray(activations), dtype=torch.float32)
    z_chunks, zpre_chunks = [], []
    for i in range(0, len(x_all), batch_size):
        z_pre, z = sae.encode(x_all[i : i + batch_size])
        z_chunks.append(z.cpu().numpy())
        zpre_chunks.append(z_pre.cpu().numpy())
    return np.concatenate(z_chunks, axis=0), np.concatenate(zpre_chunks, axis=0)


def pseudobulk_features_by_donor(
    z_features: np.ndarray,
    obs: pd.DataFrame,
    *,
    min_cells: int = 10,
    disease_positive: str = DISEASE_POSITIVE,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    # Mean feature activation per donor (the real independent unit)
    obs = obs.reset_index(drop=True)
    donor = obs["donor_id"].astype(str).to_numpy()
    disease = obs["disease"].astype(str).to_numpy()

    rows_x, rows_y, meta = [], [], []
    for d in pd.unique(donor):
        idx = np.where(donor == d)[0]
        if len(idx) < min_cells:
            continue
        if len(set(disease[idx])) != 1:
            continue
        rows_x.append(z_features[idx].mean(axis=0))
        rows_y.append(1 if disease[idx][0] == disease_positive else 0)
        meta.append({"donor_id": d, "n_cells": int(len(idx)), "disease": disease[idx][0]})

    if not rows_x:
        return (
            np.empty((0, z_features.shape[1]), dtype=float),
            np.empty((0,), dtype=int),
            pd.DataFrame(columns=["donor_id", "n_cells", "disease"]),
        )
    return np.vstack(rows_x), np.asarray(rows_y, dtype=int), pd.DataFrame(meta)


def grouped_auroc(
    feature_vec: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
) -> float:
    # Single-feature logistic AUROC under GroupKFold(groups=donor_id)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    x = np.asarray(feature_vec, dtype=float).reshape(-1, 1)
    y = np.asarray(labels, dtype=int)
    g = np.asarray(groups)
    n_groups = len(np.unique(g))
    if n_groups < 2 or len(np.unique(y)) < 2:
        return float("nan")

    gkf = GroupKFold(n_splits=min(n_splits, n_groups))
    y_true, y_score = [], []
    for tr, te in gkf.split(x, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x[tr], y[tr])
        y_true.append(y[te])
        y_score.append(clf.predict_proba(x[te])[:, 1])
    if not y_true:
        return float("nan")
    yt, ys = np.concatenate(y_true), np.concatenate(y_score)
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, ys))


def score_features_by_disease_separation(
    sae: Any,
    activations: np.ndarray,
    obs: pd.DataFrame,
    *,
    min_firing_rate: float = 0.001,
    min_cells_per_donor: int = 10,
    both_class_assays: list[str] | None = None,
    disease_positive: str = DISEASE_POSITIVE,
) -> pd.DataFrame:
    # Score live SAE features for LUAD vs normal separation
    obs = obs.reset_index(drop=True)
    y = (obs["disease"].astype(str) == disease_positive).to_numpy().astype(int)
    donor_ids = obs["donor_id"].astype(str).to_numpy()
    assays = obs["assay"].astype(str).to_numpy()
    has_counts = "n_counts" in obs.columns
    counts = obs["n_counts"].to_numpy(dtype=float) if has_counts else None
    if not has_counts:
        logger.warning("obs has no 'n_counts'; corr_total_counts will be NaN (depth gate skipped)")

    z, z_pre = encode_features(sae, activations)
    n_feat = z.shape[1]
    firing_rate = (z != 0).mean(axis=0)
    live_idx = np.where(firing_rate >= min_firing_rate)[0]
    logger.info(f"{len(live_idx)}/{n_feat} live features (firing_rate >= {min_firing_rate})")

    x_donor, y_donor, _ = pseudobulk_features_by_donor(
        z, obs, min_cells=min_cells_per_donor, disease_positive=disease_positive
    )
    logger.info(f"donor pseudobulk: {len(y_donor)} donors ({int(y_donor.sum())} LUAD)")

    both_class_assays = both_class_assays or []
    plat_mask = np.isin(assays, both_class_assays)
    plat_usable = plat_mask.sum() > 0 and len(np.unique(y[plat_mask])) == 2

    rows = []
    for f in live_idx:
        zf = z[:, f]
        zf_donor = x_donor[:, f] if len(y_donor) else np.empty(0)
        d_plat = (
            cohens_d(zf[plat_mask & (y == 1)], zf[plat_mask & (y == 0)])
            if plat_usable
            else float("nan")
        )
        rows.append(
            {
                "feature": int(f),
                "firing_rate": float(firing_rate[f]),
                "mean_z_sparse": float(zf.mean()),
                "mean_z_pre": float(z_pre[:, f].mean()),
                # cell-level (diagnostic, least trusted)
                "cohens_d_cell": cohens_d(zf[y == 1], zf[y == 0]),
                "auroc_cell": grouped_auroc(zf, y, donor_ids),
                "wilcoxon_p_cell": _ranksum_p(zf[y == 1], zf[y == 0]),
                # donor pseudobulk (primary)
                "cohens_d_donor": cohens_d(zf_donor[y_donor == 1], zf_donor[y_donor == 0])
                if len(y_donor)
                else float("nan"),
                "auroc_donor": _safe_auroc(y_donor, zf_donor) if len(y_donor) else float("nan"),
                "wilcoxon_p_donor": _ranksum_p(zf_donor[y_donor == 1], zf_donor[y_donor == 0])
                if len(y_donor)
                else float("nan"),
                # platform-controlled
                "cohens_d_within_platform": d_plat,
                # confounds
                "corr_total_counts": _safe_corr(zf, counts) if has_counts else float("nan"),
                "assay_eta_sq": _anova_eta_squared(zf, assays),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("no live features scored")
        return df
    df["wilcoxon_padj_cell"] = _bh_correct(df["wilcoxon_p_cell"])
    df["wilcoxon_padj_donor"] = _bh_correct(df["wilcoxon_p_donor"])
    return df.sort_values("cohens_d_donor", ascending=False, na_position="last").reset_index(
        drop=True
    )


def _aggregate_gene_shift(
    shift: np.ndarray, genes: np.ndarray, extractor: Any
) -> pd.DataFrame:
    # Average per-gene MVC shift across cells, keyed by gene token
    pad_id = extractor.vocab[extractor.pad_token]
    tok = genes.reshape(-1)
    val = shift.reshape(-1)
    keep = tok != pad_id
    agg = (
        pd.DataFrame({"gene_token": tok[keep], "shift": val[keep]})
        .groupby("gene_token")["shift"]
        .mean()
    )
    itos = extractor.vocab.get_itos()
    out = pd.DataFrame(
        {
            "gene_token": agg.index.astype(int),
            "gene_symbol": [itos[int(t)] for t in agg.index],
            "expected_shift": agg.to_numpy(),
        }
    )
    out["abs_shift"] = out["expected_shift"].abs()
    return out.sort_values("expected_shift", ascending=False).reset_index(drop=True)


def baseline_gepc(extractor: Any, baseline_adata: Any) -> tuple[np.ndarray, np.ndarray]:
    # Unsteered MVC output and per-cell gene tokens
    zero = np.zeros(extractor.args["embsize"], dtype=np.float32)
    return extractor.forward_with_steering(baseline_adata, zero, 0.0, return_genes=True)


def decode_feature_to_gene_weights(
    extractor: Any,
    sae: Any,
    feature_idx: int,
    baseline_adata: Any,
    *,
    alpha: float,
    layer: int | None = None,
    baseline: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    if layer is not None and layer != extractor.layer:
        logger.warning(f"decode layer={layer} != extractor.layer={extractor.layer}; using extractor.layer")
    if baseline is None:
        baseline = baseline_gepc(extractor, baseline_adata)
    mvc_base, genes = baseline

    offset = sae.decoder.weight[:, feature_idx].detach().cpu().numpy()
    mvc_steer = extractor.forward_with_steering(baseline_adata, offset, alpha)
    return _aggregate_gene_shift(mvc_steer - mvc_base, genes, extractor)


def steering_sanity_check(
    extractor: Any,
    sae: Any,
    feature_idx: int,
    baseline_adata: Any,
    decode_df: pd.DataFrame,
    *,
    alpha: float,
    baseline: tuple[np.ndarray, np.ndarray] | None = None,
    spearman_min: float = 0.3,
) -> dict[str, Any]:
    # Verify the decode by steering real baseline cells at +/- alpha
    if baseline is None:
        baseline = baseline_gepc(extractor, baseline_adata)
    mvc_base, genes = baseline
    offset = sae.decoder.weight[:, feature_idx].detach().cpu().numpy()

    plus = _aggregate_gene_shift(
        extractor.forward_with_steering(baseline_adata, offset, alpha) - mvc_base, genes, extractor
    )
    minus = _aggregate_gene_shift(
        extractor.forward_with_steering(baseline_adata, offset, -alpha) - mvc_base, genes, extractor
    )
    ref = decode_df.set_index("gene_token")["expected_shift"]
    sp = spearmanr(
        ref.to_numpy(), plus.set_index("gene_token")["expected_shift"].reindex(ref.index).to_numpy(),
        nan_policy="omit",
    ).correlation
    sm = spearmanr(
        ref.to_numpy(), minus.set_index("gene_token")["expected_shift"].reindex(ref.index).to_numpy(),
        nan_policy="omit",
    ).correlation
    return {
        "feature": int(feature_idx),
        "spearman_plus": float(sp),
        "spearman_minus": float(sm),
        "passes": bool(sp > spearman_min and sm < 0),
    }


def download_gene_sets(gmt_dir: Path, libraries: list[str] | None = None) -> None:
    # Fetch Enrichr libraries and write them as local .gmt files
    import gseapy as gp

    gmt_dir = Path(gmt_dir)
    gmt_dir.mkdir(parents=True, exist_ok=True)
    for lib in libraries or GENE_SET_LIBRARIES:
        lib_dict = gp.get_library(name=lib)
        path = gmt_dir / f"{lib}.gmt"
        with open(path, "w") as fh:
            for term, genes in lib_dict.items():
                fh.write("\t".join([term, ""] + list(genes)) + "\n")
        logger.info(f"Wrote {len(lib_dict)} gene sets to {path}")


def gsea_annotate_features(
    gene_weights: dict[int, pd.DataFrame],
    gmt_dir: Path,
    hvg_background: list[str],
    cache_dir: Path,
    *,
    n_genes: int = 100,
    libraries: list[str] | None = None,
) -> pd.DataFrame:
    import gseapy as gp

    gmt_dir = Path(gmt_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    libraries = libraries or GENE_SET_LIBRARIES
    gmts = [str(gmt_dir / f"{lib}.gmt") for lib in libraries]
    missing = [g for g in gmts if not Path(g).exists()]
    if missing:
        logger.warning(
            f"Missing GMTs {missing}; skipping GSEA. Run download_gene_sets() on a networked node."
        )
        return pd.DataFrame()

    background = list(hvg_background)
    results = []
    for fidx, gw in gene_weights.items():
        gw_sorted = gw.sort_values("expected_shift", ascending=False)
        directions = {
            "up": gw_sorted["gene_symbol"].head(n_genes).tolist(),
            "down": gw_sorted["gene_symbol"].tail(n_genes).tolist(),
        }
        for direction, genes in directions.items():
            try:
                enr = gp.enrich(gene_list=genes, gene_sets=gmts, background=background, outdir=None)
                res = enr.results.copy()
            except Exception as exc:  # noqa: BLE001 - gseapy raises broadly
                logger.warning(f"GSEA failed for feature {fidx} {direction}: {exc}")
                continue
            res.insert(0, "feature", fidx)
            res.insert(1, "direction", direction)
            res.to_csv(cache_dir / f"feature_{fidx}_{direction}.csv", index=False)
            results.append(res)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def cross_check_against_tcga_luad(
    gene_weights: dict[int, pd.DataFrame],
    tcga_csv: Path,
    hvg_genes: list[str],
    set_sizes: tuple[int, ...] = (100, 200, 500),
) -> pd.DataFrame:
    # Hypergeometric overlap of feature gene weights vs TCGA-LUAD DEGs
    tcga_csv = Path(tcga_csv)
    if not tcga_csv.exists():
        logger.warning(f"TCGA DEG file not found: {tcga_csv}; skipping. See data/external/README.md.")
        return pd.DataFrame()

    tcga = pd.read_csv(tcga_csv).dropna(subset=["gene_symbol"])
    tcga["gene_symbol"] = tcga["gene_symbol"].astype(str)
    universe = set(map(str, hvg_genes)) & set(tcga["gene_symbol"])
    m_universe = len(universe)
    tcga_in = tcga[tcga["gene_symbol"].isin(universe)]
    tcga_dirs = {
        "up": set(tcga_in.loc[tcga_in["log2fc"] > 0, "gene_symbol"]),
        "down": set(tcga_in.loc[tcga_in["log2fc"] < 0, "gene_symbol"]),
    }

    rows = []
    for fidx, gw in gene_weights.items():
        gw_in = gw[gw["gene_symbol"].astype(str).isin(universe)].sort_values(
            "expected_shift", ascending=False
        )
        for size in set_sizes:
            feat_sets = {
                "up": set(gw_in["gene_symbol"].head(size)),
                "down": set(gw_in["gene_symbol"].tail(size)),
            }
            for direction, tcga_set in tcga_dirs.items():
                feat_set = feat_sets[direction]
                k, n, big_n = len(tcga_set & feat_set), len(tcga_set), len(feat_set)
                p = (
                    float(hypergeom.sf(k - 1, m_universe, n, big_n))
                    if (m_universe and n and big_n)
                    else float("nan")
                )
                rows.append(
                    {
                        "feature": fidx,
                        "set_size": size,
                        "direction": direction,
                        "universe": m_universe,
                        "overlap_n": k,
                        "hypergeom_p": p,
                    }
                )
    return pd.DataFrame(rows)


def cell_cycle_confound_check(
    gene_weights: dict[int, pd.DataFrame],
    adata: Any,
    z_features: np.ndarray,
    *,
    ensembl_to_symbol: dict[str, str] | None = None,
    jaccard_thresh: float = 0.2,
    corr_thresh: float = 0.3,
    n_genes: int = 100,
) -> pd.DataFrame:
    # Flag features whose top genes / activations track the cell cycle
    import scanpy as sc

    a = adata.copy()
    if ensembl_to_symbol is not None:
        from sae_steering.data.gene_mapping import var_ensembl_ids

        syms = [ensembl_to_symbol.get(e) for e in var_ensembl_ids(a)]
        keep = [s is not None for s in syms]
        a = a[:, keep].copy()
        a.var_names = pd.Index([s for s in syms if s is not None]).astype(str)
        a.var_names_make_unique()

    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    s_present = [g for g in S_GENES if g in a.var_names]
    g2m_present = [g for g in G2M_GENES if g in a.var_names]
    if s_present and g2m_present:
        sc.tl.score_genes_cell_cycle(a, s_genes=s_present, g2m_genes=g2m_present)
        s_score = a.obs["S_score"].to_numpy()
        g2m_score = a.obs["G2M_score"].to_numpy()
    else:
        logger.warning("cell-cycle marker genes absent from adata; phase scores set to 0")
        s_score = g2m_score = np.zeros(a.n_obs)

    can_correlate = a.n_obs == z_features.shape[0]
    if not can_correlate:
        logger.warning(
            f"cell-cycle: adata has {a.n_obs} cells but z_features has "
            f"{z_features.shape[0]}; skipping activation-phase correlation (Jaccard only)"
        )

    g2m_set = set(G2M_GENES)
    rows = []
    for fidx, gw in gene_weights.items():
        top_pos = set(gw.sort_values("expected_shift", ascending=False)["gene_symbol"].head(n_genes))
        jac = _jaccard(top_pos, g2m_set)
        if can_correlate:
            zf = z_features[:, fidx]
            corr = max(abs(_safe_corr(zf, s_score)), abs(_safe_corr(zf, g2m_score)))
        else:
            corr = float("nan")
        rows.append(
            {
                "feature": fidx,
                "jaccard_g2m": jac,
                "cellcycle_corr": corr,
                "cellcycle_dominated": bool(jac > jaccard_thresh or corr > corr_thresh),
            }
        )
    return pd.DataFrame(rows)


def classify_feature(row: pd.Series, gates: dict[str, float] | None = None) -> str:
    # Ordered gates; default ambiguous
    g = {**DEFAULT_GATES, **(gates or {})}
    d_donor = row.get("cohens_d_donor", float("nan"))
    d_plat = row.get("cohens_d_within_platform", float("nan"))
    depth = abs(row.get("corr_total_counts") or 0.0)
    eta = row.get("assay_eta_sq") or 0.0
    cc = bool(row.get("cellcycle_dominated", False))
    firing = row.get("firing_rate", 0.0)

    same_sign = np.isfinite(d_plat) and np.isfinite(d_donor) and np.sign(d_plat) == np.sign(d_donor)
    platform_retained = np.isfinite(d_plat) and abs(d_plat) >= g["cohens_d_platform_min"] and same_sign
    confounded = depth > g["depth_corr"] or eta > g["assay_eta_sq"] or cc

    if confounded and not platform_retained:
        return "likely-confound"

    is_luad = (
        np.isfinite(d_donor)
        and abs(d_donor) >= g["cohens_d_donor_min"]
        and (row.get("wilcoxon_padj_donor", 1.0) or 1.0) < g["wilcoxon_padj_max"]
        and platform_retained
        and not confounded
        and firing >= g["min_firing_rate"]
        and (bool(row.get("gsea_hit", False)) or bool(row.get("tcga_hit", False)))
    )
    return "likely-LUAD" if is_luad else "ambiguous"
