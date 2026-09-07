#!/usr/bin/env python3
"""Print the fully resolved configuration for auditing and run manifests."""

from __future__ import annotations

import argparse

import yaml

from sonarvoxnet.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(yaml.safe_dump(load_config(args.config), sort_keys=False))


if __name__ == "__main__":
    main()
