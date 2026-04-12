# 仓库目录说明（`tensorrt-inference-optimizer`）

本文说明**各顶层目录职责**、**不应提交 Git 的内容**，以及与远程协作时的 Git 操作建议。

## 顶层目录一览

| 目录/文件 | 作用 |
|-----------|------|
| `CMakeLists.txt` | 工程构建入口：核心库、可执行文件、插件、测试注册。 |
| `README.md` / `README_EN.md` | 项目说明与快速上手（中英）。 |
| `configure.yaml` | 运行时/流水线常用配置示例（与 `cfgs/` 可二选一或并存）。 |
| `bench/` | 基准测试与引擎对比工具源码（如 `tio_benchmark`、`tio_compare_engines`）。 |
| `cfgs/` | 示例 YAML 配置（如 `bench_fp16.yaml`），供 `tio_demo` / `tio_benchmark` 使用。 |
| `checkpoint/` | **大模型资产**：两阶段 ONNX、权重等；经 **Git LFS** 跟踪（见 `.gitattributes`）。 |
| `docs/` | **人工维护**的设计、流程、架构、简历说明等文档；**不含** CI 生成的验证报告。 |
| `engine/` | **源码**：TensorRT 构建与校准器实现（`builder.*`、`calibrator.*`）；**不是** 生成引擎输出目录。 |
| `include/` | 对外头文件（如 `config.h`、`logger.h`）。 |
| `model/` | 单引擎骨架示例 ONNX 等资源（如 `bevdet.onnx`）。 |
| `plugin/` | TensorRT 自定义插件：BEVPool / Align / Gather 及注册、测试入口。 |
| `runtime/` | 推理运行时：buffer、单引擎 runner、两阶段 pipeline、BEVPool/Align CUDA 桥接。 |
| `sample0/` | 校准 batch、样例数据等**小体积**开发与测试输入。 |
| `src/` | 主程序入口（如 `demo_main.cpp`）。 |
| `tools/` | 构建/运行/导出/一键 CI/数据生成等脚本与辅助 Python。 |

## 本地生成、不应提交仓库的目录/文件

以下由 CMake 或脚本生成，**只应出现在本地或 CI 工件区**：

| 路径 | 说明 |
|------|------|
| `build/`、`build_plugin/`、`build_ci/` | CMake 构建目录；**禁止**纳入版本控制。 |
| `engine/*.engine`、`engine/*.cache` | 导出的 TensorRT 引擎与校准缓存。 |
| `reports/` | 含 `reports/ci/oneclick_ci.log`、`reports/verification_report.md` 等**生成报告**。 |
| `*.log` | 临时日志。 |

对应规则见仓库根目录 `.gitignore`。

## 与 `docs/` 的边界

- **`docs/`**：设计说明、runbook、架构、简历摘要等，**长期维护**。
- **`reports/`**：一键脚本、benchmark、量化报告等**可重复生成**的输出；路径已在 README / runbook 中说明。

## 远程仓库与 Git 操作示例

若历史上误将 `build_ci/` 等提交进仓库，在本地修正并推远程的典型流程：

```bash
cd /path/to/tensorrt-inference-optimizer

# 从索引移除构建目录（保留本地磁盘文件可加 --cached；若希望删除工作区文件则去掉 --cached）
git rm -r --cached build_ci/

# 确认 .gitignore 已包含 build_ci/ 等规则后提交
git add .gitignore docs/ tools/oneclick_ci.sh README.md README_EN.md docs/architecture.md docs/runbook.md docs/resume_project_brief.md
git status
git commit -m "chore: stop tracking build artifacts; move CI report to reports/"
```

推送到默认远程与分支：

```bash
git remote -v
git push origin HEAD
```

若使用 **Git LFS** 管理 `checkpoint/` 下大文件，克隆后需安装 LFS 并拉取指针实体：

```bash
git lfs install
git lfs pull
```

仅更新子模块或更换远程 URL（按需）：

```bash
git remote set-url origin <新的仓库 URL>
git fetch origin
git branch -u origin/master   # 若主分支名为 master
```

## 历史误提交后的协作注意

- 从 Git 删除大目录后，**仓库体积不会立刻缩小**，通常需后续 `git gc` 或在服务端执行垃圾回收；协作者需 `git pull` 后本地删除已跟踪的废弃路径。
- 若曾将 `build_ci/` 推到公开远程，建议在 PR 说明中写明「仅移除构建产物，无逻辑变更」，便于 code review。
