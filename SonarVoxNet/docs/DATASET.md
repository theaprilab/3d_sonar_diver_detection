# Dataset contract

The released repository must not contain raw sonar recordings, annotations, or
derived voxel caches unless their licenses explicitly permit redistribution.

`tools/prepare_data.py` will accept a user-supplied `--data-root` and create a
derived cache beneath that root (or an explicit `--cache-root`).  It must never
assume a laboratory-specific absolute path.

Before public release, document all of the following here:

1. The dataset access URL, license, and any access restrictions.
2. The raw directory layout expected by the adapter.
3. The annotation schema and coordinate convention.
4. The exact train, validation, and test split identifiers.
5. The deterministic preprocessing command and a cache format version.
6. Whether augmented samples are generated at training time or materialized in
   a cache.

All box rotations use the right-handed convention `R = Rz @ Ry @ Rx`, with
local axes `(length, width, height)`.  A point in local box coordinates maps to
world coordinates as `world = center + R @ local`.
