# Tahoe-100M pseudobulk drug-DE loaders (Phase 6 ground truth for Phase 7)

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import polars as pl
from loguru import logger

DE_SUBDIR = Path("metadata") / "pseudobulk_differential_expression"
GENE_VOCAB_NAME = Path("metadata") / "gene_vocabulary.json"
DE_CACHE_DIRNAME = "de_cache"
SHARD_INDEX_NAME = "cell_line_shard_index.json"
GENE_MAP_CACHE_NAME = "scgpt_gene_mapping.parquet"

DOSE_TOL = 1e-3

# Lowercased -> canonical Tahoe drug name
DRUG_ALIASES: dict[str, str] = {
    "docetaxel (trihydrate)": "Docetaxel",
    "docetaxel": "Docetaxel",
    "vinblastine (sulfate)": "Vinblastine",
    "vincristine": "Vincristine",
}


class TahoeError(Exception):
    # Base class for Tahoe loader errors.
    pass


class CellLineNotInTahoeError(TahoeError):
    # Requested cell line has no DE shard in the local snapshot.
    pass


class DrugNotScreenedError(TahoeError):
    pass


class PairNotInDEError(TahoeError):
    pass


class DoseNotFoundError(TahoeError):
    pass


def normalize_drug(name: str) -> str:
    # Strip whitespace and resolve known aliases to the canonical Tahoe name
    stripped = str(name).strip()
    return DRUG_ALIASES.get(stripped.casefold(), stripped)


def _drug_key(name: str) -> str:
    # Case-folded, alias-resolved key for matching drug names across tables.
    return normalize_drug(name).casefold()


def normalize_gene(gene_id: str) -> str:
    return str(gene_id).split(".")[0]


def _sanitize(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9.+-]+", "_", str(name))
    return cleaned.strip("_") or "x"


def _fmt_dose(concentration: float, unit: str) -> str:
    # Compact dose token for filenames, e.g
    return f"{float(concentration):g}{unit}"


def _de_dir(tahoe_dir: Path) -> Path:
    return Path(tahoe_dir) / DE_SUBDIR


def _cache_dir(tahoe_dir: Path, cache_dir: Path | None = None) -> Path:
    d = Path(cache_dir) if cache_dir is not None else Path(tahoe_dir) / DE_CACHE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_cell_line_shard_index(
    tahoe_dir: Path, *, cache_dir: Path | None = None, write_cache: bool = True
) -> dict[str, list[str]]:
    de_dir = _de_dir(tahoe_dir)
    shards = sorted(de_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no DE shards under {de_dir}")

    index: dict[str, list[str]] = {}
    for shard in shards:
        line = pl.scan_parquet(shard).select("Cell_Name_Vevo").head(1).collect().item()
        index.setdefault(str(line), []).append(shard.name)
    logger.info(
        f"built shard index: {len(index)} cell lines across {len(shards)} shards "
        f"(e.g. {max(index, key=lambda k: len(index[k]))} spans "
        f"{max(len(v) for v in index.values())} shards)"
    )

    if write_cache:
        cache_path = _cache_dir(tahoe_dir, cache_dir) / SHARD_INDEX_NAME
        cache_path.write_text(
            json.dumps({"n_shards": len(shards), "index": index}, indent=2)
        )
        logger.info(f"wrote shard index cache -> {cache_path}")
    return {line: [str(de_dir / name) for name in names] for line, names in index.items()}


def load_cell_line_shard_index(
    tahoe_dir: Path, *, rebuild: bool = False, cache_dir: Path | None = None
) -> dict[str, list[str]]:
    cache_path = _cache_dir(tahoe_dir, cache_dir) / SHARD_INDEX_NAME
    de_dir = _de_dir(tahoe_dir)
    if not rebuild and cache_path.exists():
        data = json.loads(cache_path.read_text())
        n_now = len(list(de_dir.glob("*.parquet")))
        if data.get("n_shards") == n_now:
            return {
                line: [str(de_dir / name) for name in names]
                for line, names in data["index"].items()
            }
        logger.warning(
            f"shard count changed ({data.get('n_shards')} -> {n_now}); rebuilding index"
        )
    return build_cell_line_shard_index(tahoe_dir, cache_dir=cache_dir)


def load_drug_metadata(tahoe_dir: Path) -> pd.DataFrame:
    # metadata/drug_metadata.parquet (~379 annotated drugs, MoA columns).
    return pd.read_parquet(Path(tahoe_dir) / "metadata" / "drug_metadata.parquet")


def load_sample_metadata(tahoe_dir: Path) -> pd.DataFrame:
    # metadata/sample_metadata.parquet (~1344 samples
    return pd.read_parquet(Path(tahoe_dir) / "metadata" / "sample_metadata.parquet")


def load_cell_line_metadata(tahoe_dir: Path) -> pd.DataFrame:
    # metadata/cell_line_metadata.parquet (1000 lines
    return pd.read_parquet(Path(tahoe_dir) / "metadata" / "cell_line_metadata.parquet")


def find_luad_cell_lines(
    tahoe_dir: Path,
    *,
    index: dict[str, list[str]] | None = None,
    cell_line_metadata: pd.DataFrame | None = None,
) -> list[str]:
    index = index if index is not None else load_cell_line_shard_index(tahoe_dir)
    cm = cell_line_metadata if cell_line_metadata is not None else load_cell_line_metadata(tahoe_dir)
    lung = set(cm.loc[cm["Organ"].astype(str).str.casefold() == "lung", "cell_name"].astype(str))
    return sorted(lung & set(index))


def find_drugs_by_moa(
    tahoe_dir: Path, moa_substring: str, *, drug_metadata: pd.DataFrame | None = None
) -> list[str]:
    dm = drug_metadata if drug_metadata is not None else load_drug_metadata(tahoe_dir)
    sub = str(moa_substring).casefold()
    broad = dm["moa-broad"].fillna("").astype(str).str.casefold()
    fine = dm["moa-fine"].fillna("").astype(str).str.casefold()
    mask = broad.str.contains(sub, regex=False) | fine.str.contains(sub, regex=False)
    return sorted(dm.loc[mask, "drug"].astype(str).str.strip().unique())


def _line_shards(index: dict[str, list[str]], cell_line: str) -> list[str]:
    if cell_line not in index:
        raise CellLineNotInTahoeError(
            f"{cell_line!r} has no DE shard (snapshot has {len(index)} lines with DE)"
        )
    return index[cell_line]


def line_dose_table(
    tahoe_dir: Path,
    cell_line: str,
    *,
    index: dict[str, list[str]] | None = None,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    index = index if index is not None else load_cell_line_shard_index(tahoe_dir)
    shards = _line_shards(index, cell_line)
    cache_path = _cache_dir(tahoe_dir, cache_dir) / f"{_sanitize(cell_line)}__doses.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    frames: list[pl.DataFrame] = []
    for shard in shards:
        frames.append(
            pl.scan_parquet(shard)
            .select(["drug", "concentration", "concentration_unit"])
            .unique()
            .collect()
        )
    out = (
        pl.concat(frames).unique().sort(["drug", "concentration"]).to_pandas()
        if frames
        else pd.DataFrame(columns=["drug", "concentration", "concentration_unit"])
    )
    if use_cache:
        out.to_parquet(cache_path)
    return out


def discover_available_pairs(
    tahoe_dir: Path,
    *,
    organ: str | None = "Lung",
    cell_lines: list[str] | None = None,
    drugs: list[str] | None = None,
    moa_substrings: list[str] | None = None,
    require_annotated: bool = True,
    require_in_sample_metadata: bool = True,
    index: dict[str, list[str]] | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    # Discover pullable (cell_line, drug) pairs from Tahoe itself
    index = index if index is not None else load_cell_line_shard_index(tahoe_dir)
    drug_meta = load_drug_metadata(tahoe_dir)
    sample_meta = load_sample_metadata(tahoe_dir)
    cell_meta = load_cell_line_metadata(tahoe_dir)

    # 1. candidate cell lines (must have DE).
    candidate_lines = set(index)
    if organ:
        organ_lines = set(
            cell_meta.loc[cell_meta["Organ"].astype(str).str.casefold() == organ.casefold(), "cell_name"].astype(str)
        )
        candidate_lines &= organ_lines
    if cell_lines:
        candidate_lines &= set(cell_lines)
    candidate_lines = sorted(candidate_lines)

    if drugs:
        cand_keys = {_drug_key(d) for d in drugs}
    elif moa_substrings:
        names: set[str] = set()
        for sub in moa_substrings:
            names |= set(find_drugs_by_moa(tahoe_dir, sub, drug_metadata=drug_meta))
        cand_keys = {_drug_key(d) for d in names}
    else:
        cand_keys = {_drug_key(d) for d in sample_meta["drug"].dropna().astype(str)}

    annotated_keys = {_drug_key(d) for d in drug_meta["drug"].dropna().astype(str)}
    sample_keys = {_drug_key(d) for d in sample_meta["drug"].dropna().astype(str)}
    if require_annotated:
        cand_keys &= annotated_keys
    if require_in_sample_metadata:
        cand_keys &= sample_keys

    # drug -> (moa_broad, moa_fine) lookup keyed by normalised name.
    moa_lookup: dict[str, tuple[str, str]] = {}
    for _, r in drug_meta.iterrows():
        moa_lookup.setdefault(  # first-wins -> deterministic on alias collisions
            _drug_key(r["drug"]), (str(r.get("moa-broad", "")), str(r.get("moa-fine", "")))
        )

    rows = []
    for line in candidate_lines:
        dose_tab = line_dose_table(tahoe_dir, line, index=index, cache_dir=cache_dir)
        if dose_tab.empty:
            continue
        dose_tab = dose_tab.assign(_key=dose_tab["drug"].map(_drug_key))
        for de_drug, grp in dose_tab.groupby("drug"):
            key = _drug_key(de_drug)
            if key not in cand_keys:
                continue
            concentrations = [
                {"concentration": float(c), "unit": str(u)}
                for c, u in grp[["concentration", "concentration_unit"]].itertuples(index=False)
            ]
            moa_broad, moa_fine = moa_lookup.get(key, ("", ""))
            rows.append(
                {
                    "cell_line": line,
                    "drug": str(de_drug),
                    "concentrations": concentrations,
                    "n_shards": len(index[line]),
                    "shard_paths": [Path(p).name for p in index[line]],
                    "in_sample_metadata": key in sample_keys,
                    "in_drug_metadata": key in annotated_keys,
                    "moa_broad": moa_broad,
                    "moa_fine": moa_fine,
                }
            )

    pairs = pd.DataFrame(
        rows,
        columns=[
            "cell_line", "drug", "concentrations", "n_shards", "shard_paths",
            "in_sample_metadata", "in_drug_metadata", "moa_broad", "moa_fine",
        ],
    )
    logger.info(
        f"discovered {len(pairs)} pairs across {len(candidate_lines)} candidate lines "
        f"({len(cand_keys)} candidate drugs)"
    )
    return pairs.sort_values(["cell_line", "drug"]).reset_index(drop=True)


_GENE_MAP_CACHE: dict[str, dict[str, str]] = {}
_GENEFORMER_VOCAB_CACHE: dict[str, dict[str, int]] = {}


def _symbol_to_ensembl(
    tahoe_dir: Path, gene_info_path: Path | None, cache_dir: Path | None
) -> dict[str, str]:
    if gene_info_path is not None:
        key = str(Path(gene_info_path).resolve())
        if key not in _GENE_MAP_CACHE:
            from sae_steering.data.gene_mapping import get_or_build_from_gene_info

            cache_path = _cache_dir(tahoe_dir, cache_dir) / GENE_MAP_CACHE_NAME
            df = get_or_build_from_gene_info(Path(gene_info_path), cache_path)
            mapping = (
                df.dropna(subset=["symbol", "ensembl_id"])
                .drop_duplicates(subset=["symbol"])
                .set_index("symbol")["ensembl_id"]
                .map(normalize_gene)
                .to_dict()
            )
            _GENE_MAP_CACHE[key] = mapping
        return _GENE_MAP_CACHE[key]

    gm_path = Path(tahoe_dir) / "metadata" / "gene_metadata.parquet"
    key = str(gm_path.resolve())
    if key not in _GENE_MAP_CACHE:
        gm = pd.read_parquet(gm_path)
        _GENE_MAP_CACHE[key] = (
            gm.dropna(subset=["gene_symbol", "ensembl_id"])
            .drop_duplicates(subset=["gene_symbol"])
            .set_index("gene_symbol")["ensembl_id"]
            .map(normalize_gene)
            .to_dict()
        )
    return _GENE_MAP_CACHE[key]


def _load_geneformer_vocab(tahoe_dir: Path) -> dict[str, int]:
    # Tahoe Geneformer gene vocabulary: version-stripped Ensembl -> int index.
    path = str((Path(tahoe_dir) / GENE_VOCAB_NAME).resolve())
    if path not in _GENEFORMER_VOCAB_CACHE:
        raw = json.loads(Path(path).read_text())
        _GENEFORMER_VOCAB_CACHE[path] = {normalize_gene(k): int(v) for k, v in raw.items()}
    return _GENEFORMER_VOCAB_CACHE[path]


def load_pseudobulk_de(
    tahoe_dir: Path,
    cell_line: str,
    drug: str,
    *,
    concentration: float = 0.5,
    concentration_unit: str = "uM",
    dose: tuple[float, str] | None = None,
    use_cache: bool = True,
    gene_info_path: Path | None = None,
    index: dict[str, list[str]] | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    # Standardised pseudobulk DE for one (cell_line, drug, dose)
    if dose is not None:
        concentration, concentration_unit = float(dose[0]), str(dose[1])
    index = index if index is not None else load_cell_line_shard_index(tahoe_dir)
    shards = _line_shards(index, cell_line)  # raises CellLineNotInTahoeError

    cache_path = (
        _cache_dir(tahoe_dir, cache_dir)
        / f"{_sanitize(cell_line)}_{_sanitize(drug)}_{_fmt_dose(concentration, concentration_unit)}.parquet"
    )
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    # Resolve the requested drug to DE spelling(s) on this line
    dose_tab = line_dose_table(tahoe_dir, cell_line, index=index, cache_dir=cache_dir)
    de_drugs = dose_tab["drug"].astype(str)
    req = str(drug).strip().casefold()
    spellings = sorted({d for d in de_drugs if d.strip().casefold() == req})
    if not spellings:
        want = _drug_key(drug)
        spellings = sorted({d for d in de_drugs if _drug_key(d) == want})
        if len(spellings) > 1:
            logger.warning(
                f"{drug!r} alias-matches multiple DE spellings {spellings} on "
                f"{cell_line!r}; pulling only {spellings[0]!r} (pass an exact spelling for another)"
            )
            spellings = spellings[:1]
    if not spellings:
        sample = sorted(de_drugs.unique())[:8]
        raise PairNotInDEError(
            f"drug {drug!r} not screened on {cell_line!r} "
            f"({de_drugs.nunique()} drugs available, e.g. {sample})"
        )

    avail = dose_tab[dose_tab["drug"].isin(spellings)]
    dose_hit = avail[
        ((avail["concentration"].astype(float) - float(concentration)).abs() <= DOSE_TOL)
        & (avail["concentration_unit"].astype(str) == concentration_unit)
    ]
    if dose_hit.empty:
        doses = sorted(
            {(round(float(c), 4), str(u)) for c, u in avail[["concentration", "concentration_unit"]].itertuples(index=False)}
        )
        raise DoseNotFoundError(
            f"{cell_line}/{drug}: {concentration}{concentration_unit} not found; available: {doses}"
        )

    # Pull gene-level rows for the matched (drug, dose) -- per shard, frugal.
    frames: list[pl.DataFrame] = []
    for shard in shards:
        df = (
            pl.scan_parquet(shard)
            .filter(
                pl.col("drug").is_in(spellings)
                & ((pl.col("concentration") - float(concentration)).abs() <= DOSE_TOL)
                & (pl.col("concentration_unit") == concentration_unit)
            )
            .collect()
        )
        if df.height:
            frames.append(df)
    if not frames:
        raise DoseNotFoundError(
            f"{cell_line}/{drug}: no DE rows for {concentration}{concentration_unit} "
            "(dose-table cache may be stale; rebuild de_cache)"
        )
    raw = pl.concat(frames).to_pandas()

    sym2ens = _symbol_to_ensembl(tahoe_dir, gene_info_path, cache_dir)
    out = pd.DataFrame(
        {
            "gene_id": raw["gene_name"].map(sym2ens),
            "gene_symbol": raw["gene_name"].astype(str),
            "log2_fold_change": raw["log2FoldChange"].astype("float64"),
            "pvalue": raw["pvalue"].astype("float64"),
            "padj": raw["padj"].astype("float64"),
            "mean_expression": raw["baseMean"].astype("float64"),
            "concentration": raw["concentration"].astype("float64"),
            "concentration_unit": raw["concentration_unit"].astype(str),
            "cell_line": str(cell_line),
            "drug": raw["drug"].astype(str),
        }
    )
    if use_cache:
        out.to_parquet(cache_path)
    return out


def _vocab_symbol_token(scgpt_vocab: object) -> tuple[set, object]:
    if hasattr(scgpt_vocab, "get_stoi"):  # scGPT GeneVocab
        stoi = scgpt_vocab.get_stoi()
        return set(stoi), stoi.get
    if isinstance(scgpt_vocab, dict):
        return set(scgpt_vocab), scgpt_vocab.get
    return set(map(str, scgpt_vocab)), None  # set/list of symbols, no token ids


def align_genes_with_scgpt_vocab(
    de_df: pd.DataFrame, scgpt_vocab: object
) -> tuple[pd.DataFrame, dict]:
    # Keep DE genes whose symbol is in the scGPT vocab; report dropout stats
    symbols_set, token_of = _vocab_symbol_token(scgpt_vocab)
    in_vocab = de_df["gene_symbol"].astype(str).isin(symbols_set)
    aligned = de_df[in_vocab].copy()
    if token_of is not None:
        aligned["gene_token"] = aligned["gene_symbol"].astype(str).map(token_of)

    n_input = int(len(de_df))
    n_aligned = int(in_vocab.sum())
    aligned_no_ens = int(aligned["gene_id"].isna().sum()) if "gene_id" in aligned else 0
    stats = {
        "n_input": n_input,
        "n_aligned": n_aligned,
        "n_dropout_symbol": int((~in_vocab).sum()),
        "n_dropout_no_ensembl": aligned_no_ens,
        "dropout_rate": float(1 - n_aligned / n_input) if n_input else 0.0,
    }
    return aligned.reset_index(drop=True), stats


def align_genes_with_geneformer_vocab(
    de_df: pd.DataFrame,
    tahoe_dir: Path,
    *,
    gene_info_path: Path | None = None,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    # Keep DE genes whose Ensembl id is in the Tahoe Geneformer vocab
    vocab = _load_geneformer_vocab(tahoe_dir)
    df = de_df.copy()
    if "gene_id" not in df.columns or df["gene_id"].isna().all():
        sym2ens = _symbol_to_ensembl(tahoe_dir, gene_info_path, cache_dir)
        df["gene_id"] = df["gene_symbol"].astype(str).map(sym2ens)

    ens = df["gene_id"].map(lambda g: normalize_gene(g) if isinstance(g, str) else g)
    has_ens = ens.notna()
    in_vocab = ens.map(lambda g: g in vocab if isinstance(g, str) else False)
    aligned = df[in_vocab].copy()
    aligned["gene_id"] = ens[in_vocab]
    aligned["geneformer_index"] = aligned["gene_id"].map(vocab.get)

    n_input = int(len(df))
    n_aligned = int(in_vocab.sum())
    stats = {
        "n_input": n_input,
        "n_aligned": n_aligned,
        "n_dropout_no_ensembl": int((~has_ens).sum()),
        "n_dropout_symbol": int((has_ens & ~in_vocab).sum()),
        "dropout_rate": float(1 - n_aligned / n_input) if n_input else 0.0,
    }
    return aligned.reset_index(drop=True), stats


# Prompt 6/7 paper examples that are absent from the local Tahoe snapshot
ABSENT_BENCHMARK_DRUGS: tuple[str, ...] = ("Cisplatin", "Carboplatin")
ABSENT_BENCHMARK_LINES: tuple[str, ...] = ("NCI-H1299", "NCI-H1975", "H1975")


def parse_manifest_concentrations(value: object) -> list[dict]:
    # Parse the concentrations JSON column from pair_manifest.parquet.
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def select_doses_for_manifest_row(
    concentrations: list[dict],
    *,
    pull_all_concentrations: bool = False,
    default_concentration: float = 0.5,
    default_concentration_unit: str = "uM",
) -> list[tuple[float, str]]:
    if not concentrations:
        return []
    if pull_all_concentrations:
        return [(float(c["concentration"]), str(c["unit"])) for c in concentrations]
    target = float(default_concentration)
    unit = str(default_concentration_unit)
    same_unit = [c for c in concentrations if str(c["unit"]) == unit]
    pool = same_unit or concentrations
    best = min(pool, key=lambda c: abs(float(c["concentration"]) - target))
    return [(float(best["concentration"]), str(best["unit"]))]


def signature_parquet_path(
    sig_dir: Path,
    cell_line: str,
    drug: str,
    concentration: float,
    unit: str,
    *,
    vocab: str = "scgpt",
) -> Path:
    # Path to a Phase 6 aligned signature parquet for one (line, drug, dose).
    tag = f"{_sanitize(cell_line)}__{_sanitize(drug)}__{_fmt_dose(concentration, unit)}"
    return Path(sig_dir) / f"{tag}_{vocab}.parquet"


def resolve_signature_paths(
    manifest: pd.DataFrame,
    sig_dir: Path,
    *,
    vocab: str = "scgpt",
    pull_all_concentrations: bool = False,
    default_concentration: float = 0.5,
    default_concentration_unit: str = "uM",
) -> list[Path]:
    sig_dir = Path(sig_dir)
    paths: list[Path] = []
    seen: set[Path] = set()
    skipped_rows = 0

    for _, row in manifest.iterrows():
        doses = select_doses_for_manifest_row(
            parse_manifest_concentrations(row["concentrations"]),
            pull_all_concentrations=pull_all_concentrations,
            default_concentration=default_concentration,
            default_concentration_unit=default_concentration_unit,
        )
        row_paths: list[Path] = []
        for conc, unit in doses:
            path = signature_parquet_path(
                sig_dir, row["cell_line"], row["drug"], conc, unit, vocab=vocab
            )
            if path.exists():
                row_paths.append(path)

        if not row_paths and not pull_all_concentrations:
            prefix = f"{_sanitize(row['cell_line'])}__{_sanitize(row['drug'])}__"
            suffix = f"_{vocab}.parquet"
            row_paths = sorted(sig_dir.glob(f"{prefix}*{suffix}"))

        if not row_paths:
            skipped_rows += 1
            continue
        for path in row_paths:
            if path not in seen:
                paths.append(path)
                seen.add(path)

    logger.info(
        f"resolve_signature_paths: {len(paths)} {vocab} files from "
        f"{len(manifest)} manifest rows"
        + (f"; {skipped_rows} rows had no on-disk signature" if skipped_rows else "")
    )
    return paths


def benchmark_example_status(
    *,
    tahoe_dir: Path | None = None,
    manifest: pd.DataFrame | None = None,
    shard_index: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    # Status of Prompt 6/7 example drugs/lines (expected absences documented).
    manifest = manifest if manifest is not None else pd.DataFrame()
    haystack = (
        " ".join(
            manifest.get(col, pd.Series(dtype=str)).astype(str).str.cat(sep=" ")
            for col in ["cell_line", "drug"]
        ).casefold()
        if not manifest.empty
        else ""
    )

    sample_keys: set[str] | None = None
    cell_names: set[str] | None = None
    if tahoe_dir is not None:
        sample_keys = {_drug_key(d) for d in load_sample_metadata(tahoe_dir)["drug"].dropna()}
        cell_names = set(load_cell_line_metadata(tahoe_dir)["cell_name"].astype(str))

    out: dict[str, str] = {}
    for drug in ABSENT_BENCHMARK_DRUGS:
        if drug.casefold() in haystack:
            out[drug] = "present in pair_manifest"
        elif sample_keys is not None and _drug_key(drug) in sample_keys:
            out[drug] = "screened in Tahoe but not in filtered manifest"
        else:
            out[drug] = "absent from Tahoe (not screened)"

    for line in ABSENT_BENCHMARK_LINES:
        if line.casefold() in haystack:
            out[line] = "present in pair_manifest"
        elif shard_index is not None and line in shard_index:
            out[line] = "has DE shard but not in filtered manifest"
        elif cell_names is not None and line in cell_names:
            out[line] = "absent from pull (no DE shard)"
        else:
            out[line] = "absent from Tahoe (not in metadata)"
    return out
