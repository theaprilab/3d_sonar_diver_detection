# Reproducibility contract

Every paper number must map to one committed YAML configuration, an exact
command, and a result file.  Do not reconstruct a reported experiment from
undocumented command-line overrides.

For each run, `tools/train.py` must save the resolved YAML configuration,
random seed, package versions, CUDA details, git revision, and command line in
the run directory and checkpoint metadata.  `tools/evaluate.py` must write a
machine-readable JSON report next to its human-readable summary.

The release checklist is:

- [ ] Pin the tested Python, PyTorch, CUDA, and spconv versions.
- [ ] Publish the exact data preparation command and split files.
- [ ] Add one command per main result and ablation row.
- [ ] Release checkpoints or provide a documented download location.
- [ ] Map every paper table and figure to an output JSON or plotting script.
- [ ] Run `pytest` and one end-to-end smoke test in a clean environment.
- [ ] Select an open-source license and replace the draft license notice.

The public training and evaluation path must not depend on laboratory-specific
infrastructure, credentials, or filesystem paths.
