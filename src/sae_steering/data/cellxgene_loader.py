# CellxGene Census loader for LUAD + normal lung cells

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from loguru import logger

# Pin the Census version for reproducibility
DEFAULT_CENSUS_VERSION = "2025-01-30"

# Exact CellxGene ontology label
DISEASE_LABELS = {
    "luad": "lung adenocarcinoma",
    "normal": "normal",
}

LUAD_OBS_COLUMNS = [
    "soma_joinid",
    "disease",
    "donor_id",
    "assay",
    "cell_type",
    "tissue",
    "tissue_general",
]

ALLOWED_LUAD_DISEASES = [DISEASE_LABELS["luad"], DISEASE_LABELS["normal"]]


def _disease_normal_value_filter(disease_label: str, tissue_general: str) -> str:
    # Census obs filter for balanced <disease> + healthy <tissue_general>.
    normal = DISEASE_LABELS["normal"]
    return (
        f"tissue_general == '{tissue_general}' and is_primary_data == True and "
        f"(disease == '{disease_label}' or disease == '{normal}')"
    )


def _luad_normal_value_filter() -> str:
    # Census obs filter for balanced LUAD + healthy lung (back-compat shim).
    return _disease_normal_value_filter(DISEASE_LABELS["luad"], "lung")


def _ensure_soma_joinid_column(
    obs_column_names: list[str] | None,
) -> list[str]:
    columns = list(obs_column_names or LUAD_OBS_COLUMNS)
    if "soma_joinid" not in columns:
        columns.insert(0, "soma_joinid")
    return columns


def _pick_soma_joinids(
    obs_df: pd.DataFrame,
    n_cells: int,
    seed: int,
    *,
    balance: bool = False,
    balance_labels: list[str] | None = None,
) -> np.ndarray:
    # Sample Census soma_joinid values without loading expression data
    if "soma_joinid" not in obs_df.columns:
        raise ValueError("obs_df must include a 'soma_joinid' column")

    rng = np.random.default_rng(seed)
    all_joinids = obs_df["soma_joinid"].to_numpy()

    if balance:
        if balance_labels is None:
            raise ValueError("balance_labels required when balance=True")
        per_group = n_cells // 2
        chunks: list[np.ndarray] = []
        for label in balance_labels:
            group_ids = obs_df.loc[obs_df["disease"] == label, "soma_joinid"].to_numpy()
            if len(group_ids) == 0:
                logger.warning(f"No cells found for disease='{label}'")
                continue
            take = min(per_group, len(group_ids))
            if take < per_group:
                logger.warning(
                    f"Only {len(group_ids):,} cells available for '{label}' "
                    f"(requested {per_group:,})"
                )
            chunks.append(rng.choice(group_ids, size=take, replace=False))
        if not chunks:
            raise RuntimeError("No cells matched any balance_labels group")
        picked = np.concatenate(chunks)
        rng.shuffle(picked)
        return picked

    take = min(n_cells, len(all_joinids))
    if len(all_joinids) > take:
        return rng.choice(all_joinids, size=take, replace=False)
    return all_joinids


def _filter_allowed_diseases(
    adata: ad.AnnData,
    allowed: list[str],
) -> ad.AnnData:
    # Drop cells whose disease label is outside the requested set.
    mask = adata.obs["disease"].isin(allowed).values
    n_drop = int((~mask).sum())
    if n_drop:
        dropped = adata.obs.loc[~mask, "disease"].value_counts().to_dict()
        logger.warning(
            f"Dropping {n_drop} cells outside allowed diseases {allowed}: {dropped}"
        )
        adata = adata[mask].to_memory()
    return adata


def _fetch_census_cells(
    n_cells: int,
    *,
    value_filter: str,
    census_version: str,
    organism: str = "homo_sapiens",
    seed: int = 0,
    balance: bool = False,
    balance_labels: list[str] | None = None,
    obs_column_names: list[str] | None = None,
) -> ad.AnnData:
    import cellxgene_census

    obs_column_names = _ensure_soma_joinid_column(obs_column_names)

    logger.info(
        f"Querying Census obs metadata (v={census_version!r}, "
        f"filter={value_filter!r}, cap={n_cells:,})"
    )

    with cellxgene_census.open_soma(census_version=census_version) as census:
        obs_df = cellxgene_census.get_obs(
            census,
            organism,
            value_filter=value_filter,
            column_names=obs_column_names,
        )
        if obs_df.empty:
            raise RuntimeError(
                f"Census obs query returned 0 cells for filter: {value_filter}"
            )

        picked = _pick_soma_joinids(
            obs_df,
            n_cells,
            seed,
            balance=balance,
            balance_labels=balance_labels,
        )
        logger.info(
            f"Fetching expression for {len(picked):,} / {len(obs_df):,} candidate cells"
        )
        adata = cellxgene_census.get_anndata(
            census,
            organism=organism,
            obs_coords=picked.tolist(),
            obs_column_names=obs_column_names,
        )

    return adata.to_memory()


def load_census_slice(
    n_cells: int,
    *,
    tissue: str = "lung",
    organism: str = "homo_sapiens",
    census_version: str = "stable",
    disease: str | None = None,
    seed: int = 0,
    obs_column_names: list[str] | None = None,
) -> ad.AnnData:
    filters = [f"tissue_general == '{tissue}'", "is_primary_data == True"]
    if disease is not None:
        filters.append(f"disease == '{disease}'")
    value_filter = " and ".join(filters)

    return _fetch_census_cells(
        n_cells,
        value_filter=value_filter,
        census_version=census_version,
        organism=organism,
        seed=seed,
        obs_column_names=obs_column_names,
    )


def _subsample_adata(
    adata: ad.AnnData,
    n_cells: int,
    seed: int,
    *,
    balance: bool = False,
    balance_labels: list[str] | None = None,
) -> ad.AnnData:
    # Subsample cells, optionally balancing across balance_labels.
    if adata.n_obs <= n_cells:
        return adata
    rng = np.random.default_rng(seed)
    if balance and balance_labels:
        per_label = max(1, n_cells // len(balance_labels))
        picks: list[int] = []
        for label in balance_labels:
            idx = np.where(adata.obs["disease"].astype(str).to_numpy() == label)[0]
            if len(idx) == 0:
                continue
            take = min(per_label, len(idx))
            picks.extend(rng.choice(idx, size=take, replace=False).tolist())
        if not picks:
            picks = rng.choice(adata.n_obs, size=min(n_cells, adata.n_obs), replace=False).tolist()
        idx = np.sort(np.unique(picks))
        if len(idx) > n_cells:
            idx = np.sort(rng.choice(idx, size=n_cells, replace=False))
        return adata[idx].copy()
    idx = np.sort(rng.choice(adata.n_obs, size=n_cells, replace=False))
    return adata[idx].copy()


def load_disease_and_normal(
    disease_label: str,
    tissue_general: str,
    n_cells: int = 50_000,
    balance: bool = True,
    census_version: str = DEFAULT_CENSUS_VERSION,
    cache_path: Path | None = None,
    seed: int = 0,
) -> ad.AnnData:
    allowed = [disease_label, DISEASE_LABELS["normal"]]
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        logger.info(f"Loading cached AnnData from {cache_path}")
        adata = ad.read_h5ad(cache_path)
        adata = _filter_allowed_diseases(adata, allowed)
        if adata.n_obs < n_cells:
            logger.warning(
                f"Cached h5ad has {adata.n_obs:,} cells < requested {n_cells:,}; "
                "re-querying Census"
            )
        else:
            if adata.n_obs > n_cells:
                logger.info(
                    f"Cached h5ad has {adata.n_obs:,} cells; subsampling to {n_cells:,}"
                )
                adata = _subsample_adata(
                    adata,
                    n_cells,
                    seed,
                    balance=balance,
                    balance_labels=allowed if balance else None,
                )
            return adata

    logger.info(
        f"Querying CellxGene Census v{census_version} for "
        f"{disease_label!r} + normal {tissue_general} cells (cap {n_cells:,})"
    )

    adata = _fetch_census_cells(
        n_cells,
        value_filter=_disease_normal_value_filter(disease_label, tissue_general),
        census_version=census_version,
        seed=seed,
        balance=balance,
        balance_labels=allowed if balance else None,
        obs_column_names=LUAD_OBS_COLUMNS,
    )

    adata = _filter_allowed_diseases(adata, allowed)
    logger.info(f"Census returned {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    _log_batch_distribution(adata)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Caching AnnData to {cache_path}")
        adata.write_h5ad(cache_path)

    return adata


def load_luad_and_normal_lung(
    n_cells: int = 50_000,
    balance: bool = True,
    census_version: str = DEFAULT_CENSUS_VERSION,
    cache_path: Path | None = None,
    seed: int = 0,
) -> ad.AnnData:
    return load_disease_and_normal(
        disease_label=DISEASE_LABELS["luad"],
        tissue_general="lung",
        n_cells=n_cells,
        balance=balance,
        census_version=census_version,
        cache_path=cache_path,
        seed=seed,
    )


def _log_batch_distribution(adata: ad.AnnData) -> None:
    # Print donor and assay counts so batch confounds are visible.
    disease_counts = pd.Series(adata.obs["disease"].astype(str)).value_counts().to_dict()
    logger.info(f"Disease counts: {disease_counts}")

    n_donors_per_disease = (
        adata.obs.groupby("disease", observed=True)["donor_id"].nunique().to_dict()
    )
    logger.info(f"Unique donors per disease: {n_donors_per_disease}")

    for disease, n_donors in n_donors_per_disease.items():
        if n_donors < 3:
            logger.warning(
                f"Only {n_donors} unique donor(s) for disease='{disease}'. "
                f"SAE features may pick up donor identity instead of biology."
            )

    assay_counts = adata.obs["assay"].value_counts().head(5).to_dict()
    logger.info(f"Top assays: {assay_counts}")
