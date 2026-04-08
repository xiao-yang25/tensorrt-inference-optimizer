#!/usr/bin/env python3
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser("Generate dummy INT8 calibration batches")
    parser.add_argument("--out-dir", default="sample0")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--shape", default="1,6,3,256,704")
    args = parser.parse_args()

    shape = tuple(int(x) for x in args.shape.split(","))
    np.random.seed(7)
    for i in range(args.count):
        arr = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        path = f"{args.out_dir}/calib_batch_{i:03d}.bin"
        arr.tofile(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
