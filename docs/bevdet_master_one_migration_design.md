# BEVDet `master/one` 能力迁移到 Optimizer 的设计方案

## 1. 目标与范围

目标：把 `/data/bevdet-tensorrt-cpp` 的成熟实现（`master` 与 `one` 分支）有计划地迁移/适配到 `tensorrt-inference-optimizer`，并结合较新的推理与量化优化方法，形成可持续迭代的部署框架。

范围包含：

- 推理链路：从“单 engine scaffold”升级到“多阶段/插件融合可切换”
- 算子能力：BEVPool、预处理、时序对齐、GatherBEV
- 部署性能：低时延与高吞吐并行优化
- 量化部署：INT8 为主，预留 FP8 路径

不在本阶段范围：

- 重新训练模型（仅定义 QAT/PTQ 接口和校验协议）
- 业务上游数据标注/评测平台改造

---

## 2. 现状差异分析（基于本机仓库）

### 2.1 `bevdet-tensorrt-cpp` 分支差异结论

`origin/master -> origin/one` 文件级差异显示：

- `master`：双阶段引擎 + 中间 CUDA kernel（`preprocess.cu`、`bevpool.cu`、`grid_sampler.cu`）
- `one`：插件化和单引擎方向（新增 `preprocess_plugin`、`bevpool_plugin`、`alignbev_plugin`、`gatherbev_plugin`，并把 `tools/export_engine.py` 重命名为 `tools/export_one_engine.py`）

核心代码证据：

- `master` 流程为 STEP1~STEP6（预处理 -> image stage -> bev pool -> align -> bev stage -> 后处理）
- `one` 的 `DoInfer` 只调用一次 `trt_context->executeV2(...)`，说明中间能力被合并到单 engine 图中

### 2.2 `tensorrt-inference-optimizer` 当前状态

- 目前偏“工程骨架”：
  - 运行时：`runtime/infer.*` 是通用单 engine runner
  - plugin：已具备 `BevPool` / `AlignBev` / `GatherBev` 可运行实现与分层测试；后续可继续向 `IPluginV3` 与单引擎全图融合演进
  - 文档和脚本已支持两阶段 ONNX 的引擎构建与独立 benchmark

---

## 3. 目标架构（建议采用“双轨并行”）

## Track-A（近期落地）：双阶段可生产链路

对齐 `master` 思路，优先保证可交付：

1. `ImgStageEngine`（TensorRT）
2. 中间 CUDA 模块（BEVPool + 可选 Align）
3. `BEVStageEngine`（TensorRT）
4. 后处理（NMS/解码）

优点：

- 风险低，调试边界清晰
- 容易复用你本地 `checkpoint/` 下的 `img_stage*.onnx`、`bev_stage*.onnx`（需自备，不入库）
- 容易做阶段级 profiling（可直接定位瓶颈）

## Track-B（中期优化）：单引擎插件融合链路

参考 `one` 思路，把中间模块逐步 plugin 化并并入主 engine：

- PreprocessPlugin
- BEVPoolPlugin
- AlignBEVPlugin（时序）
- GatherBEVPlugin

优点：

- 减少 host 端调度和 kernel launch overhead
- 缓解中间 tensor 读写拷贝
- 端到端时延更稳定（尤其高帧率场景）

风险：

- 插件维护成本高、跨 TensorRT 版本兼容成本高
- 数值一致性验证难度更高

---

## 4. 迁移矩阵（从 `bevdet-tensorrt-cpp` 到 Optimizer）

| 能力模块 | 来源分支 | 迁移优先级 | 目标落点（optimizer） | 说明 |
|---|---|---:|---|---|
| 双 engine 配置与运行 | master | P0 | `cfgs/` + `runtime/` | 新增 `multi_stage_runner`，支持 `img/bev` 两 context |
| BEVPool CUDA kernel | master/one | P0 | `plugin/` 或 `runtime/kernels/` | 先 kernel 版可用，再 plugin 版替换 |
| 时序特征对齐（adj） | master/one | P1 | `runtime/temporal/` + plugin | 可先保留 C++/CUDA 实现，后续 plugin 化 |
| 预处理融合（resize/crop/norm） | one | P1 | `preprocess/` + plugin | 保留 CPU fallback |
| GatherBEV | one | P2 | plugin | 与时序策略耦合，后置 |
| 单引擎导出工具链 | one | P2 | `tools/export_one_engine.py` | 与 Track-B 一起启用 |

---

## 5. 分阶段实施计划

## Phase 0：统一接口与配置（1 周）

交付：

- 新配置结构：支持 `pipeline_mode: two_stage | one_engine`
- `configure.yaml` 拆分为：
  - `build.img_engine` / `build.bev_engine` / `build.one_engine`
  - `runtime.temporal.use_adj`、`runtime.preprocess.mode`
- `InferRunner` 抽象为 `IInferencePipeline`

验收：

- 旧单 engine 配置不破
- 两阶段配置可加载并完成 smoke test

## Phase 1：落地 Track-A 双阶段生产链路（2~3 周）

交付：

- 新增 `runtime/two_stage_pipeline.*`
- 集成 `BEVPool` kernel（先不用 plugin）
- 对齐 `master` 的阶段计时口径：`pre/img/bevpool/align/bev/post/total`

验收指标：

- 功能：在真实 checkpoint ONNX + 引擎下可跑通
- 性能：具备稳定阶段耗时报告
- 正确性：与基线输出误差在可接受阈值内（定义 per-head MAE/IoU）

## Phase 2：插件化（Track-B）与单引擎模式（3~5 周）

交付：

- 插件库：`tio_plugins` 扩展为 preprocess/bevpool/align/gather 四插件
- 新增 `tools/export_one_engine.py`（或 C++ builder）用于构建单 engine
- 支持 `one_engine` 推理模式及回退策略（自动回退 two_stage）

验收指标：

- 单引擎相比双阶段：
  - 目标时延改善：5%~20%（视平台 CPU-bound 程度）
  - 数值一致性通过（关键 head 误差阈值）

## Phase 3：量化优化（2~4 周）

交付：

- INT8 PTQ pipeline（校准集管理 + cache 管理 + 精度报告）
- Q/DQ 显式量化导入路径（优先兼容 TensorRT 10）
- 预留 FP8 评估开关（硬件支持时启用）

验收指标：

- `FP16 -> INT8` 时延收益与精度折衷可量化
- 引擎 cache/timing cache 可复用，构建时长显著下降

---

## 6. 推理优化策略（较新实践，适配到本项目）

## 6.1 TensorRT 构建与运行

1. **Timing Cache 持久化**
   - 对同平台/同模型复用 tactic 选择结果，降低重复 build 成本
2. **CUDA Graph（仅在 CPU-bound 时启用）**
   - 对稳定 shape、固定执行序列场景减少 launch overhead
3. **多 Profile + 固定主 profile**
   - 实际部署选择最常见 shape 作为 `opt`，减少 tactic 抖动
4. **减少 Host-Device 往返**
   - 中间 tensor 常驻 GPU，后处理尽量 GPU 化

参考：

- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/10.12.0/performance/best-practices.html)

## 6.2 量化部署

1. **INT8 PTQ 基线**
   - 先做 per-tensor/per-channel 标准校准，建立收益和精度基线
2. **显式 Q/DQ 图（推荐中期）**
   - 降低隐式量化不确定性，便于跨版本复现
3. **FP8 路径（硬件允许时）**
   - 在 Ada/Hopper 及后续平台可尝试，优先在大吞吐场景评估
4. **QAT 作为高精度兜底**
   - 当 PTQ 精度不可接受时启用，成本更高但更稳

参考：

- [TensorRT Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [TensorRT Model Optimizer: Choosing Quant Methods](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_choosing_quant_methods.html)

## 6.3 跨框架部署可选路径

- 引入 ONNX Runtime TensorRT EP 作为回归与兼容路径（非主路径）：
  - 优点：切换成本低、cache 配置方便
  - 用途：快速验证 TRT subgraph 性能和兼容性

参考：

- [ONNX Runtime TensorRT EP](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)

---

## 7. 风险与规避

1. **TensorRT 版本碎片化（8.x / 10.x）**
   - 规避：封装版本适配层（API、builder flags、plugin 注册）
2. **插件 ABI/序列化兼容问题**
   - 规避：插件版本号和序列化 schema 强约束，增加反序列化回归测试
3. **量化精度回退**
   - 规避：构建层级白名单（敏感层保持 FP16/FP32）+ QAT 兜底
4. **多流并发导致时序缓存错用**
   - 规避：按 scene/token 管理 temporal buffer 生命周期

---

## 8. 建议里程碑（建议版本号）

- `v0.2`: 双阶段 pipeline 正式可用（替代仅单 engine scaffold）
- `v0.3`: 插件化能力上线（至少 BEVPool + Align）
- `v0.4`: 单引擎模式可选上线（灰度）
- `v0.5`: INT8 量化生产可用，提供精度-性能报告模板

---

## 9. 与 BEVDet 导出分支关系

`BEVDet-export` 分支导出的 `img_stage*.onnx` / `bev_stage*.onnx` 本质上就是为 `bevdet-tensorrt-cpp` 这类分段部署准备。  
因此 Optimizer 的设计不应只绑定“一个 onnx = 一个 engine”，而应支持：

- 两阶段（高可维护）
- 单引擎（高性能上限）
- 可回退（生产稳定性）

参考：

- [LCH1238/BEVDet export README_zh-CN](https://github.com/LCH1238/BEVDet/blob/export/README_zh-CN.md)
