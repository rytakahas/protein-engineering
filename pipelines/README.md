# Pipelines (Orchestration)

These scripts are *thin wrappers* around the package CLIs you already have under `packages/`. They read a single YAML config and run each stage in order. You can pass `--dry-run` to print the commands instead of executing them.

## 0) Prepare your environment
- Install your packages in editable mode or ensure the scripts under `packages/*/scripts/` are runnable:
  ```bash
  pip install -e packages/rescontact
  pip install -e packages/resintnet
  pip install -e packages/seqml
  ```
  *(Alternatively, these pipeline scripts simply spawn the Python CLIs under `packages/.../scripts/` via `subprocess`.)*

## 1) End-to-end proposal (structure → hotspots → mutants → seq ML)
```bash
python pipelines/e2e_propose_mutations.py   --config configs/pipeline.example.yaml   --dry-run   # remove to actually run
```

The stages:
1. **ResContact**: homologs → template priors, ESM2 embeddings, MSA features
2. **ResIntNet**: build graph, compute PRS, run GNN, blend → ranked residues
3. **SeqML**: enumerate mutants at top-K residues and fine-tune/evaluate sequence model

## 2) Train individual modules
```bash
python pipelines/train_rescontact.py --config configs/pipeline.example.yaml --dry-run
python pipelines/train_resintnet.py  --config configs/pipeline.example.yaml --dry-run
python pipelines/train_seqml.py      --config configs/pipeline.example.yaml --dry-run
```

Each script reads the same config and assembles sensible commands to your existing CLIs.
