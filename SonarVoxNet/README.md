# SonarVoxNet

This repository is the publication-oriented codebase for SonarVoxNet.  It is
being prepared independently from the exploratory research workspace; only the
paper model and stated ablations belong here.

## Scope

The default experiment is the proposed model:

`VFE -> sparse 3D convolution -> BEV backbone -> center head -> 6D rotation head`

It predicts a Gaussian center heatmap, center offset, height, log dimensions,
and a continuous 6D rotation representation.  Rotation supervision is L1 on
the encoded target.  The sparse middle encoder uses `spconv`.

The following configurations are retained solely for the paper ablations:

| Study | Configurations |
| --- | --- |
| Middle encoder | Dense Conv3D, sparse Conv3D (proposed) |
| Detection head | Anchor, center with z-yaw target, center with full 3D target |
| Rotation representation | Quaternion, axis-angle, 6D, 9D |

## Repository status

The package and experiment contracts are scaffolded.  The next migration step
is to port the reviewed data adapter, target generation, training loop, and
evaluator from the exploratory workspace into this repository.  No result in
the paper should claim reproducibility from this draft until the commands in
`docs/REPRODUCIBILITY.md` are marked complete.

## Layout

```text
configs/             Versioned paper and ablation recipes
src/sonarvoxnet/models/
                     VFE, sparse encoder, BEV backbone, and heads
src/sonarvoxnet/training/
                     Losses, trainer, and run-manifest utilities
src/sonarvoxnet/evaluation/
                     Metrics, prediction export, and evaluation protocol
tools/               Thin public command-line entry points
docs/                Dataset and reproducibility contracts
tests/               Fast mathematical and smoke tests
```

## Environment

Python 3.10+ and a CUDA-compatible PyTorch build are required for training.
Install the project and the sparse-convolution extra that matches the local CUDA
runtime:

```bash
pip install -e '.[sparse,dev]'
```

`spconv` wheels are CUDA-version specific.  Record the exact PyTorch, CUDA,
and spconv versions used for every released checkpoint.

## Intended commands

The commands below are the stable public interface.  They will become runnable
when the data adapter migration is complete.

```bash
python tools/prepare_data.py --config configs/main_spconv_6d_l1.yaml --data-root /path/to/data
python tools/train.py --config configs/main_spconv_6d_l1.yaml --data-root /path/to/data --output-dir outputs/main_s0
python tools/evaluate.py --config configs/main_spconv_6d_l1.yaml --data-root /path/to/data --checkpoint checkpoints/sonarvoxnet_main.pt
```

See [dataset instructions](docs/DATASET.md) and the
[reproducibility contract](docs/REPRODUCIBILITY.md).
