# sae-steering

Core library for sparse autoencoders on scGPT activations, with steering validation against Tahoe-100M drug signatures.

## Install

```bash
pip install -e .
pip install --no-deps git+https://github.com/bowang-lab/scGPT.git@main
```

Optional extras:

```bash
pip install -e ".[geneformer,flash,tcga,viz,logging]"
```

scGPT must be installed with `--no-deps` because its pins conflict with torch 2.3+.
