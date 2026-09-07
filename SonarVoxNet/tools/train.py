#!/usr/bin/env python3
"""Public training entry point for a resolved experiment configuration."""

from __future__ import annotations

import argparse

from sonarvoxnet.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SonarVoxNet from a paper recipe.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config["model"]["middle_encoder"] != "sparse":
        print("Running an ablation configuration.")
    raise SystemExit("Training-loop migration is not complete. This draft intentionally does not run hidden experimental code.")


if __name__ == "__main__":
    main()
