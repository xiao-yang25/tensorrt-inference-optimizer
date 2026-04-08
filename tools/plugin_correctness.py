#!/usr/bin/env python3
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser("Simple plugin numerical checker")
    parser.add_argument("--input", required=True, help="Path to input npy")
    parser.add_argument("--output", required=True, help="Path to plugin output npy")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    x = np.load(args.input)
    y = np.load(args.output)
    if x.shape != y.shape:
      raise SystemExit(f"Shape mismatch: {x.shape} vs {y.shape}")

    ok = np.allclose(x, y, atol=args.atol, rtol=args.rtol)
    max_abs = float(np.max(np.abs(x - y)))
    print(f"allclose={ok}, max_abs_diff={max_abs}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
