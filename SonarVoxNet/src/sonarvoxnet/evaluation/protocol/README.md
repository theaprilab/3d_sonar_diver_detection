# Evaluation protocol

This directory is the single source of truth for paper evaluation rules.
The final release must place the following versioned artifacts here:

1. The 3D oriented-box IoU implementation and its numerical tolerances.
2. AP matching thresholds and interpolation convention.
3. The prediction JSON schema and a minimal valid example.
4. The split identifier used for each paper table.
5. Rotation metrics: full SO(3) geodesic error and any reported yaw-only metric.
6. A command that maps one checkpoint to one result JSON.

The evaluator must read a fixed YAML recipe and write all thresholds, split
names, checkpoint hashes, and package versions into its output.  It must not
calibrate thresholds on the test split.
