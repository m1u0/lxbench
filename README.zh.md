# LXBench

[English](README.md) | 中文

LXBench 是一个极简、模块化的 harness，用于针对已经运行的推理端点执行可复现、以代码评分的 LLM 基准测试。每个基准测试自行负责准备和评分，而一个小型、与基准测试无关的执行层负责运行它们。

Python runner 仅使用标准库。Python 不可用时，Bash fallback 只需要 `curl` 和少量基础 shell 工具，无需 JSON 工具或基准测试依赖。

## 设计

基准测试专属代码位于 `benchmarks/` 下各个自包含目录中。每个目录包含其准备器、评分器、依赖、文档、测试及所需的 provider 文件。共享边界是一份小型文件约定，而不是 plugin system 或基准测试框架。

LXBench 采用 provider-first 原则：在可能的情况下，评分遵循基准测试的提供方。每个基准测试 README 都记录其来源、请求配置以及所有必要的偏离之处。添加基准测试不应要求修改共享 runners。

## 工作流程

1. 基准测试专属的准备器下载并缓存官方数据集，然后写入请求、稳定 ID 和评分用例。
2. 共享 runner 将这些请求原样发送到 OpenAI-compatible chat-completions endpoint，并保存原始响应。
3. 基准测试专属的评分器将响应与用例关联，并报告该基准测试的标准指标。

准备好的数据可多次运行。原始响应与分数分开保存，因此无需重复推理即可重新评分。

## 基准测试

| 基准测试 | 评估内容 | 报告内容 |
| --- | --- | --- |
| [MMLU-Redux 2.0](benchmarks/mmlu_redux/README.md) | 多项选择知识与推理 | 总体准确率与各学科准确率 |
| [IFEval](benchmarks/ifeval/README.md) | 指令遵循 | 严格和宽松的提示级与指令级准确率 |
| [LongBench v2](benchmarks/longbench_v2/README.md) | 长上下文理解 | 按难度和上下文长度细分的总体准确率 |

各基准测试的 README 说明其依赖、提供方来源、请求配置、评分细节，以及相对提供方推理设置的偏离。

## 快速开始：MMLU-Redux

工作站流程需要 Python 3.10 或更高版本，以及一个已经运行的推理 endpoint。安装 MMLU-Redux 依赖：

```sh
python3 -m pip install -r benchmarks/mmlu_redux/requirements.txt
```

准备数据集：

```sh
python3 benchmarks/mmlu_redux/prepare.py \
  --cache-dir data/mmlu-redux-2.0 \
  --output prepared/mmlu-redux-2.0
```

针对 endpoint 运行：

```sh
python3 run.py \
  --endpoint http://board.example:8080/v1/chat/completions \
  --requests prepared/mmlu-redux-2.0/requests.jsonl \
  --output results/mmlu-redux-2.0/raw.jsonl \
  --concurrency 4
```

对保存的响应评分：

```sh
python3 benchmarks/mmlu_redux/grade.py \
  --cases prepared/mmlu-redux-2.0/cases.jsonl \
  --responses results/mmlu-redux-2.0/raw.jsonl \
  --output results/mmlu-redux-2.0/score.json
```

评分器会打印结果，并将完整的 JSON 摘要写入输出路径。

## 准备所有基准测试

安装每个基准测试 README 所述的依赖后，可使用以下命令准备所有当前数据集：

```sh
python3 prepare.py --longbench-context-size 262144
```

LongBench 上下文大小必须与目标 server 配置的有效上下文一致。已有的有效输出会被复用。传入 `--force` 可重新生成输出，包括在更改 LongBench 上下文大小后。

## 运行与恢复

两个 runner 均支持并发、可复现采样、瞬时失败重试，以及按稳定 case ID 恢复。成功响应会立即写入。复用输出路径时会跳过已完成的 ID 并重试缺失项；使用新路径会启动一次独立运行。

添加 `--sample-size N`，并可选择添加 `--seed N`，即可运行稳定子集。采样运行会写入 `<output>.manifest.json`。请将此文件与原始响应放在一起，以便评分器恢复所选分母，包括未产生响应的请求。

## 在精简系统上运行

若工作站无法访问 endpoint，请将 `run.sh` 以及准备好的 `requests.jsonl` 和 `ids.txt` 文件复制到目标系统，然后运行：

```sh
./run.sh \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --requests requests.jsonl \
  --output raw.jsonl \
  --concurrency 4
```

将 `raw.jsonl` 复制回工作站进行评分；若为采样运行，还需复制其 manifest。该 fallback 需要 Bash、`curl`、`mkdir`、`rm` 和 `sleep`。它没有 JSON parser，因此 HTTP 2xx 响应正文必须是单行紧凑 JSON。工作站评分器会执行完整的响应验证。

## 添加基准测试

每个准备器发布三个对齐的 UTF-8 文件：

| 文件 | 用途 |
| --- | --- |
| `requests.jsonl` | runner 原样发送的完整 JSON 请求正文 |
| `ids.txt` | 每个请求一个稳定 ID，按行对齐 |
| `cases.jsonl` | 用于评分的基准测试专属信息 |

runner 仅读取请求和 ID。评分器按 ID 将用例与原始响应关联。新基准测试在 `benchmarks/` 下提供准备器、评分器、依赖列表、README 和聚焦测试。必要时，提供方代码位于精简的 `upstream/` 目录中。

该约定适用于请求可独立运行的确定性基准测试。交互式或依赖响应的基准测试需要不同的执行模型。

## 范围

LXBench 评估一个已经运行的 endpoint。它不会下载或加载模型、管理推理 server、添加认证、比较 endpoints、汇总基准测试分数、判定回归，也不提供专用的延迟测量。

## 保持 README 同步

每当任一 README 发生更改时，请在同一项更改中更新其对应版本。
