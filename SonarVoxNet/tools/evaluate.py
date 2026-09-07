#!/usr/bin/env python3
"""Public evaluation entry point for one checkpoint and one fixed recipe."""

from __future__ import annotations

import argparse

from sonarvoxnet.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SonarVoxNet with the paper metric protocol.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    load_config(args.config)
    raise SystemExit("Evaluator migration is not complete. This draft intentionally does not publish provisional metrics.")


if __name__ == "__main__":
    main()
