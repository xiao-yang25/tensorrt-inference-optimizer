# Performance Report Template

## Build Info

- Commit:
- GPU:
- Driver:
- CUDA / TensorRT:
- Engine mode: FP32 / FP16 / INT8
- Profile: min / opt / max

## Accuracy Delta

| Mode | Metric | Value |
|---|---|---|
| FP16 vs FP32 | mAP delta | |
| INT8 vs FP16 | mAP delta | |

## Runtime Metrics

| Mode | Mean(ms) | P50(ms) | P90(ms) | P99(ms) | Throughput(samples/s) |
|---|---:|---:|---:|---:|---:|
| FP32 | | | | | |
| FP16 | | | | | |
| INT8 | | | | | |

## Observed Bottlenecks

- Kernel:
- Memory bandwidth:
- H2D/D2H overlap:

## Optimization Decisions

- Stream count:
- Pinned memory strategy:
- Workspace MB:
- Plugin route enabled:
