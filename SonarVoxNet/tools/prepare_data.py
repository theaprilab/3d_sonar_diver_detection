#!/usr/bin/env python3
"""Public data-preparation entry point.

The project-specific adapter is deliberately not copied from the exploratory
workspace.  This command reserves the stable interface and prevents hidden
laboratory paths from becoming part of the public API.
"""

from __future__ import annotations

import argparse

from sonarvoxnet.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a versioned SonarVoxNet cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", default=None)
    args = parser.parse_args()
    load_config(args.config)
    raise SystemExit("Data adapter migration is not complete. See docs/DATASET.md before using this draft.")


if __name__ == "__main__":
    main()
