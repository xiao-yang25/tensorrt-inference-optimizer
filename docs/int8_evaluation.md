# INT8 Evaluation Protocol

## Goal

Evaluate INT8 acceleration against FP16 baseline while controlling acceptable accuracy drop.

## Procedure

1. Generate or collect representative calibration batches.
2. Build FP16 engine and record benchmark.
3. Build INT8 engine with calibrator cache.
4. Run identical benchmark settings and compare.
5. Run detection metric script on the same validation split.

### Practical commands (current repo)

```bash
# Fast path: one script
scripts/run_ptq_baseline.sh

# Or manual path:
# 1) Build binaries that include benchmark + engine comparator
cmake -S . -B build -DTIO_ENABLE_BENCHMARK=ON
cmake --build build -j --target tio_benchmark tio_compare_engines

# 2) Prepare engines (example names)
#    - FP16: output/engines/bevdet_fp16.engine
#    - INT8: output/engines/bevdet_int8.engine

# 3) Generate one-shot report (trtexec performance + output diff)
python scripts/int8_fp16_report.py \
  --fp16-engine output/engines/bevdet_fp16.engine \
  --int8-engine output/engines/bevdet_int8.engine \
  --compare-bin build/tio_compare_engines \
  --json-out output/reports/int8_fp16_report.json
```

`tio_compare_engines` uses identical synthetic inputs for both engines and reports:

- `overall_mae`
- `overall_rmse`
- `overall_max_abs`
- `overall_mismatched/overall_count`

## Required Outputs

- `output/cache/calib.cache` generated and reused.
- `output/engines/bevdet_int8.engine` and `output/engines/bevdet_fp16.engine`.
- Latency/throughput comparison table.
- Accuracy deltas for key metrics (for example mAP and NDS).

## Acceptance Gate

- INT8 engine build succeeds with cache hit on second build.
- Throughput gain is measurable versus FP16.
- Accuracy drop stays within project threshold.
