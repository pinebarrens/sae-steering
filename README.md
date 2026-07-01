# Externally Validating Steered Disease Features in Single-Cell Foundation Models

A reproducible pipeline for scGPT that discovers disease-associated SAE features, applies an internal steering validity check, and compares the steered gene shifts to drug-induced expression in Tahoe-100M. We train sparse autoencoders on scGPT representations of lung adenocarcinoma and invasive ductal breast carcinoma cells, identify directions that separate tumors from normal cells, steer healthy cells along those axes, and test the induced shifts against measured cancer cell-line drug response.

## Install

Use a virtual environment so scGPT's dependency pins do not collide with the rest of the stack:

```bash
pip install -e .
pip install --no-deps git+https://github.com/bowang-lab/scGPT.git@main
```

scGPT must be installed with `--no-deps` because its pins conflict with torch 2.3+. Optional extras enable individual stages: `flash`, `geneformer`, `tcga`, `viz`, `logging`.

```bash
pip install -e ".[geneformer,flash,tcga,viz,logging]"
```

## Data and checkpoints

CELLxGENE Census cells are downloaded automatically. Two assets you supply, passed to each stage as path arguments:

- **scGPT whole-human checkpoint** (`args.json`, `vocab.json`, `best_model.pt`, `gene_info.csv`) from the [scGPT](https://github.com/bowang-lab/scGPT) release. Build the Ensembl→symbol map once with `data.gene_mapping.build_mapping_from_gene_info`.
- **Tahoe-100M snapshot** from the [Tahoe-100M](https://doi.org/10.1101/2025.02.20.639398) release, with the per-cell-line DE shards and drug/sample/cell-line metadata under `data/tahoe/metadata/`.

## Pipeline

Each stage is a plain module under `src/sae_steering/` that can be run and cached on its own.

1. **Sample cells** — `data.cellxgene_loader.load_disease_and_normal` samples 50,000 cells balanced between malignant and normal states within the matched tissue.
2. **Extract activations** — `models.scgpt_wrapper.scGPTActivationExtractor` averages the gene-token activations at layer 6 of the 12-layer model into one vector per cell.
3. **Train the SAE** — `training.train_sae.SAETrainer` trains a Top-K sparse autoencoder (8,192 features, K=32 active per cell, unit-norm decoder columns) on the cached activations.
4. **Discover features** — `analysis.feature_discovery` ranks features by Cohen's *d* between malignant and normal cells on per-donor mean activations, keeping only those that separate the two within a single platform and are not driven by sequencing platform, library size, or cell-cycle score.
5. **Check steering validity** — `analysis.steering.validity_check_steering` offsets healthy cells along a feature's decoder direction and requires that their projection onto the malignant–normal axis rise monotonically with strength (Spearman ρ ≥ 0.8).
6. **Compare to drug response** — `analysis.drug_comparison.full_hypothesis_test` scores each validated feature by Spearman correlation over at least 500 shared genes against Tahoe pseudobulk drug DE, tests against a null of 100 random decoder directions, and applies Benjamini–Hochberg correction within each cohort.

```python
import numpy as np
from sae_steering.data.cellxgene_loader import load_disease_and_normal
from sae_steering.models.scgpt_wrapper import scGPTActivationExtractor
from sae_steering.models.sae import TopKSAE
from sae_steering.training.train_sae import SAETrainer

adata = load_disease_and_normal("lung adenocarcinoma", "lung", n_cells=50_000,
                                cache_path="data/cache/luad_lung_50k.h5ad")

extractor = scGPTActivationExtractor("data/scgpt/whole_human", "data/gene_mapping.parquet", layer=6)
np.save("data/cache/luad_layer6.npy", extractor.extract_activations(adata))

sae = TopKSAE(d_input=512, d_latent=8192, k=32)
SAETrainer(sae, "data/cache/luad_layer6.npy", "data/cache/sae_luad").train()
```

The comparison is cross-context, since steering acts on patient-tissue cells while Tahoe uses cancer cell lines. It benchmarks three drugs of differing mechanism per cohort — docetaxel, oxaliplatin, and erlotinib across five lung lines, and paclitaxel, palbociclib, and tucatinib on two breast lines.

## Extending

Every stage is disease-parameterized: point `load_disease_and_normal` at any malignant/normal pair in CELLxGENE and set `disease_positive` in the discovery call. `models.geneformer_wrapper.GeneformerActivationExtractor` mirrors the scGPT wrapper's API for Geneformer V2-104M.

## Bugs & Suggestions

Please report any bugs, problems, suggestions, or requests as a [GitHub issue](https://github.com/pinebarrens/sae-steering/issues).
