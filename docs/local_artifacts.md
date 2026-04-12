# 本地产物说明（不入库）

以下路径**不会**出现在远程仓库中，克隆后需在本地自行准备（或运行脚本生成）。

## 目录约定

| 路径 | 用途 |
|------|------|
| `checkpoint/` | 两阶段 ONNX、权重等；按 [BEVDet export 说明](https://github.com/LCH1238/BEVDet/blob/export/README_zh-CN.md) 导出后放入此目录（与 `tools/run_checkpoint_trt.sh` 默认路径一致）。 |
| `model/` | 单引擎骨架用 ONNX（如 `bevdet.onnx`）；自行放入或从上游发布物获取。 |
| `sample0/` | INT8 校准 batch、BEVPool 索引、可选 `input.npy`/`output.npy` 等；可用本仓库脚本生成。 |
| `engine/*.engine`、`engine/*.cache` | TensorRT 引擎与校准缓存；由 `tools/export_engine.py` 或 `tio_demo` 构建阶段生成。 |

## 常用生成命令

```bash
mkdir -p sample0

# INT8 校准用随机 batch（骨架验证）
python tools/generate_dummy_calib.py --out-dir sample0 --count 8 --shape 1,6,3,256,704

# BEVPool 索引（需外部相机 YAML 与模型配置）
python tools/generate_bevpool_indices.py \
  --model-cfg /path/to/bevdet.yaml \
  --cam-yaml /path/to/sample0000.yaml \
  --out-dir sample0/bevpool_indices
```

## 配置

`cfgs/default.yaml` 与 `configure.yaml` 中的 `build.onnx_path`、`build.calibration_batches`、`two_stage.*_engine_path` 等均为**相对于仓库根的路径示例**；请按你本地实际文件位置修改。
