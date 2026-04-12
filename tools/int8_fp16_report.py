#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict


def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def parse_trtexec_metrics(output: str) -> Dict[str, float]:
    metrics = {}
    m = re.search(r"GPU Compute Time:.*mean = ([0-9.]+) ms", output)
    if m:
        metrics["gpu_mean_ms"] = float(m.group(1))
    m = re.search(r"Latency:.*mean = ([0-9.]+) ms", output)
    if m:
        metrics["latency_mean_ms"] = float(m.group(1))
    m = re.search(r"Throughput: ([0-9.]+) qps", output)
    if m:
        metrics["throughput_qps"] = float(m.group(1))
    return metrics


def parse_compare_metrics(output: str) -> Dict[str, float]:
    metrics = {}
    keys = ["overall_mae", "overall_rmse", "overall_max_abs"]
    for k in keys:
        m = re.search(rf"{k}=([0-9.eE+-]+)", output)
        if m:
            metrics[k] = float(m.group(1))
    m = re.search(r"overall_mismatched=([0-9]+)/([0-9]+)", output)
    if m:
        metrics["overall_mismatched"] = int(m.group(1))
        metrics["overall_count"] = int(m.group(2))
    return metrics


def benchmark_engine(trtexec_bin: str, engine: str, warmup_ms: int, iterations: int) -> Dict[str, float]:
    out = run_cmd(
        [
            trtexec_bin,
            f"--loadEngine={engine}",
            f"--warmUp={warmup_ms}",
            f"--iterations={iterations}",
            "--duration=0",
            "--avgRuns=10",
        ]
    )
    metrics = parse_trtexec_metrics(out)
    if not metrics:
        raise RuntimeError(f"Failed to parse trtexec metrics from {engine}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Generate FP16 vs INT8 report")
    ap.add_argument("--fp16-engine", required=True)
    ap.add_argument("--int8-engine", required=True)
    ap.add_argument("--compare-bin", default="build/tio_compare_engines")
    ap.add_argument("--trtexec-bin", default="trtexec")
    ap.add_argument("--warmup-ms", type=int, default=200)
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--json-out", default="reports/int8_fp16_report.json")
    args = ap.parse_args()

    if not os.path.exists(args.fp16_engine):
        raise FileNotFoundError(args.fp16_engine)
    if not os.path.exists(args.int8_engine):
        raise FileNotFoundError(args.int8_engine)
    if not os.path.exists(args.compare_bin):
        raise FileNotFoundError(args.compare_bin)

    fp16 = benchmark_engine(args.trtexec_bin, args.fp16_engine, args.warmup_ms, args.iterations)
    int8 = benchmark_engine(args.trtexec_bin, args.int8_engine, args.warmup_ms, args.iterations)
    compare_out = run_cmd([args.compare_bin, args.fp16_engine, args.int8_engine])
    err = parse_compare_metrics(compare_out)

    report = {
        "fp16": fp16,
        "int8": int8,
        "error": err,
        "derived": {},
    }
    if "latency_mean_ms" in fp16 and "latency_mean_ms" in int8:
        report["derived"]["latency_speedup"] = fp16["latency_mean_ms"] / int8["latency_mean_ms"]
    if "throughput_qps" in fp16 and "throughput_qps" in int8:
        report["derived"]["throughput_gain"] = int8["throughput_qps"] / fp16["throughput_qps"]

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n## FP16 vs INT8 Report")
    print(f"- FP16 latency mean(ms): {fp16.get('latency_mean_ms', 'N/A')}")
    print(f"- INT8 latency mean(ms): {int8.get('latency_mean_ms', 'N/A')}")
    print(f"- Latency speedup: {report['derived'].get('latency_speedup', 'N/A')}")
    print(f"- FP16 throughput(qps): {fp16.get('throughput_qps', 'N/A')}")
    print(f"- INT8 throughput(qps): {int8.get('throughput_qps', 'N/A')}")
    print(f"- Throughput gain: {report['derived'].get('throughput_gain', 'N/A')}")
    print(f"- overall_mae: {err.get('overall_mae', 'N/A')}")
    print(f"- overall_rmse: {err.get('overall_rmse', 'N/A')}")
    print(f"- overall_max_abs: {err.get('overall_max_abs', 'N/A')}")
    print(f"- mismatched: {err.get('overall_mismatched', 'N/A')}/{err.get('overall_count', 'N/A')}")
    print(f"- JSON: {args.json_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
