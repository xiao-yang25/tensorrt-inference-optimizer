#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate identity 3x3 transforms for temporal geometric align."
    )
    parser.add_argument("--adj-num", type=int, required=True, help="Number of adjacent frames")
    parser.add_argument("--out", required=True, help="Output float32 binary file path")
    return parser.parse_args()


def main():
    args = parse_args()
    mats = np.zeros((args.adj_num, 3, 3), dtype=np.float32)
    for i in range(args.adj_num):
        mats[i] = np.eye(3, dtype=np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(mats.reshape(-1).tobytes())
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
