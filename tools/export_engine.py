#!/usr/bin/env python3
import argparse
import subprocess
import sys


def build_cmd(args):
    cmd = [
        "trtexec",
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        f"--workspace={args.workspace}",
        "--verbose",
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.int8:
        cmd.append("--int8")
    if args.min_shapes and args.opt_shapes and args.max_shapes:
        cmd.append(f"--minShapes={args.min_shapes}")
        cmd.append(f"--optShapes={args.opt_shapes}")
        cmd.append(f"--maxShapes={args.max_shapes}")
    if args.calib:
        cmd.append(f"--calib={args.calib}")
    return cmd


def main():
    parser = argparse.ArgumentParser("TensorRT engine exporter")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--workspace", type=int, default=4096)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--calib", default="")
    parser.add_argument("--min-shapes", default="")
    parser.add_argument("--opt-shapes", default="")
    parser.add_argument("--max-shapes", default="")
    args = parser.parse_args()

    cmd = build_cmd(args)
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    sys.exit(rc)


if __name__ == "__main__":
    main()
