# INT8 Evaluation Protocol

## Goal

Evaluate INT8 acceleration against FP16 baseline while controlling acceptable accuracy drop.

## Procedure

1. Generate or collect representative calibration batches.
2. Build FP16 engine and record benchmark.
3. Build INT8 engine with calibrator cache.
4. Run identical benchmark settings and compare.
5. Run detection metric script on the same validation split.

## Required Outputs

- `engine/calib.cache` generated and reused.
- `engine/bevdet_int8.engine` and `engine/bevdet_fp16.engine`.
- Latency/throughput comparison table.
- Accuracy deltas for key metrics (for example mAP and NDS).

## Acceptance Gate

- INT8 engine build succeeds with cache hit on second build.
- Throughput gain is measurable versus FP16.
- Accuracy drop stays within project threshold.
