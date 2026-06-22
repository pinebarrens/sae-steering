# Ensembl gene ID -> HGNC symbol mapping

from __future__ import annotations

from pathlib import Path

import pandas as pd
from anndata import AnnData
from loguru import logger

_MIN_CACHE_ENTRIES = 1000


def var_ensembl_ids(adata: AnnData) -> list[str]:
    # Return Ensembl IDs for each gene column
    if "feature_id" in adata.var.columns:
        return (
            adata.var["feature_id"]
            .astype(str)
            .str.split(".")
            .str[0]
            .tolist()
        )
    return adata.var.index.astype(str).tolist()


def build_mapping(ensembl_ids: list[str], cache_path: Path) -> pd.DataFrame:
    # Query mygene.info for Ensembl -> symbol and cache to parquet
    import mygene

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    mg = mygene.MyGeneInfo()
    logger.info(f"Querying mygene for {len(ensembl_ids)} Ensembl IDs (one-time)")

    results = mg.querymany(
        ensembl_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        returnall=False,
    )

    rows = []
    n_missing = 0
    for r in results:
        if r.get("notfound") or "symbol" not in r:
            n_missing += 1
            continue
        rows.append({"ensembl_id": r["query"], "symbol": r["symbol"]})

    df = pd.DataFrame(rows).drop_duplicates(subset=["ensembl_id"]).reset_index(drop=True)
    df.to_parquet(cache_path)

    logger.info(
        f"Cached {len(df)} Ensembl -> symbol mappings to {cache_path} "
        f"({n_missing} of {len(ensembl_ids)} IDs unresolved)"
    )
    return df


def load_mapping(cache_path: Path) -> pd.DataFrame:
    # Load the cached mapping
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Gene mapping cache not found: {cache_path}\n"
            f"Build it once with sae_steering.data.gene_mapping.build_mapping(...)"
        )
    return pd.read_parquet(cache_path)


def get_or_build(ensembl_ids: list[str], cache_path: Path) -> pd.DataFrame:
    # Load the cached mapping if it exists, otherwise build it from mygene.
    cache_path = Path(cache_path)
    if cache_path.exists():
        return load_mapping(cache_path)
    return build_mapping(ensembl_ids, cache_path)


def build_mapping_from_gene_info(gene_info_path: Path, cache_path: Path) -> pd.DataFrame:
    # Build Ensembl -> symbol mapping from scGPT's gene_info.csv.
    gene_info_path = Path(gene_info_path)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(gene_info_path)
    if {"feature_id", "feature_name"}.issubset(raw.columns):
        df = raw[["feature_id", "feature_name"]].rename(
            columns={"feature_id": "ensembl_id", "feature_name": "symbol"}
        )
    elif {"gene_id", "gene_name"}.issubset(raw.columns):
        df = raw[["gene_id", "gene_name"]].rename(
            columns={"gene_id": "ensembl_id", "gene_name": "symbol"}
        )
    else:
        raise ValueError(
            f"{gene_info_path} must have feature_id/feature_name or "
            f"gene_id/gene_name columns; got {list(raw.columns)}"
        )

    df["ensembl_id"] = df["ensembl_id"].astype(str).str.split(".").str[0]
    df["symbol"] = df["symbol"].astype(str)
    df = df.drop_duplicates(subset=["ensembl_id"]).reset_index(drop=True)
    df.to_parquet(cache_path)

    logger.info(
        f"Cached {len(df)} Ensembl -> symbol mappings from {gene_info_path} "
        f"to {cache_path}"
    )
    return df


def get_or_build_from_gene_info(gene_info_path: Path, cache_path: Path) -> pd.DataFrame:
    gene_info_path = Path(gene_info_path)
    cache_path = Path(cache_path)

    if cache_path.exists():
        df = load_mapping(cache_path)
        if len(df) >= _MIN_CACHE_ENTRIES:
            logger.info(f"Loaded gene mapping ({len(df)} entries) from {cache_path}")
            return df
        logger.warning(
            f"Gene mapping cache has only {len(df)} entries (expected >="
            f"{_MIN_CACHE_ENTRIES}); rebuilding from {gene_info_path}"
        )

    if not gene_info_path.exists():
        raise FileNotFoundError(
            f"Gene info file not found: {gene_info_path}\n"
            "Download it from the scGPT whole_human checkpoint."
        )
    return build_mapping_from_gene_info(gene_info_path, cache_path)
