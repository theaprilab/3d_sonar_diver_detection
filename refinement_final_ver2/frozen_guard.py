"""Fail-closed integrity check for the accepted geometry implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "frozen_geometry_checksums.json"


def verify_frozen_sources(root: Path = HERE) -> dict:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mismatches = {}
    for relative, expected in payload["algorithm_files"].items():
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(f"frozen geometry integrity check failed: {mismatches}")
    return {
        "snapshot": payload["snapshot"],
        "verified_files": len(payload["algorithm_files"]),
        "verified": True,
    }


if __name__ == "__main__":
    print(json.dumps(verify_frozen_sources(), indent=2))

