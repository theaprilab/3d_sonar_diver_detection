# Benchmark protocol branch

Branch name: `benchmark/protocol`

This branch contains a framework-independent benchmark package in
`SonarVoxNet/benchmark/`. It includes the protocol, common configuration,
evaluator, prediction-export template, and minimal example prediction and
ground-truth files.

The benchmark package should remain independent of PyTorch, spconv, and any
model implementation. It is the canonical definition of prediction schema, 3D
OBB IoU, AP matching, score thresholds, and split identifiers. Keep the
protocol versioned and do not calibrate any setting on the test split.
