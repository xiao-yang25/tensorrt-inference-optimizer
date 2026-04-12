# 仓库目录说明

## 原则

- **只提交源码与文档**：`include/`、`src/`、`engine/`、`runtime/`、`plugin/`（其下按类型分为 `include/`、`src/`、`cuda/`、`tests/`）、`bench/`、`scripts/`、`cfgs/`、`configure.yaml`、`docs/`、`CMakeLists.txt` 等。
- **CMake 生成物**：统一在仓库根目录 **`build/`**（日常开发默认 `-B build`；一键 CI 使用 **`build/ci`** 子目录；Runbook 中插件单测示例使用 **`build/plugin`**）。
- **运行产出**：统一在 **`output/`**（`engines/`、`cache/`、`reports/` 等），已 `.gitignore`。
- **本地数据**：`checkpoint/`、`model/`、`sample0/` 仅本地使用，不入库。

## `output/` 子目录约定

| 路径 | 说明 |
|------|------|
| `output/engines/` | TensorRT `.engine` |
| `output/cache/` | INT8 校准缓存等 |
| `output/reports/` | 验证报告、CI 日志、量化 JSON 报告 |

配置中的相对路径示例见 `configure.yaml` 与 `cfgs/default.yaml`。

## `.gitignore` 摘要

忽略 `build/`、`output/`、`checkpoint/`、`model/`、`sample0/`、`__pycache__/`、`engine/*.engine`、`reports/`（历史路径）等。
