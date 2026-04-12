#!/usr/bin/env python3
import argparse
import subprocess
import sys


def detect_workspace_flag():
    """Pick compatible workspace flag by inspecting trtexec help."""
    try:
        result = subprocess.run(
            ["trtexec", "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        return "--workspace"
    output = result.stdout or ""
    if "--memPoolSize" in output:
        return "--memPoolSize"
    return "--workspace"


def build_cmd(args):
    workspace_flag = detect_workspace_flag()
    cmd = [
        "trtexec",
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        "--verbose",
    ]
    if workspace_flag == "--memPoolSize":
        cmd.append(f"--memPoolSize=workspace:{args.workspace}")
    else:
        cmd.append(f"--workspace={args.workspace}")
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
