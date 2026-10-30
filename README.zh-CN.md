# macMLX

[English](README.md) · **简体中文**

> 由 Apple MLX 驱动的原生 macOS 本地大模型推理工具。

macMLX 让 Apple Silicon 以头等原生体验跑本地 LLM——无云、无遥测、无
Electron。给新手一个精致的 SwiftUI 应用，给开发者一把趁手的 CLI，再给
其它一切一个常驻的 OpenAI 兼容 API。

---

## 为什么选 macMLX？

MLX 推理和 CLI 曾经就是全部卖点——但截至 2026，
[LM Studio](https://github.com/lmstudio-ai/mlx-engine) 和
[Ollama](https://ollama.com/blog/mlx) 在 Apple Silicon 上都上了 MLX 引擎，
LM Studio 也有 `lms` CLI。所以诚实的对比是关于**组合**：真正原生的 macOS
GUI、常驻 API、零 Python，全在一个约 50 MB 的 app 里。

| | macMLX | LM Studio | Ollama | oMLX |
|--|--------|-----------|--------|------|
| 原生 macOS GUI | ✅ SwiftUI | Electron | 仅菜单栏 | ✅ SwiftUI（v0.4+） |
| **Swift 原生进程内引擎** | ✅ | ❌ | ❌ | ❌（Python 内核） |
| MLX 推理 | ✅ | ✅ | ✅（预览） | ✅ |
| 命令行 (CLI) | ✅ | ✅ `lms` | ✅ | 仅启动器 |
| 断点续传 + 镜像源 | ✅ | ⚠ 部分 | ⚠ 部分 | ❌ |
| OpenAI 兼容 API | ✅ 常驻 | ✅ | ✅ | ✅ |
| 无需 Python | ✅ | ✅ | ✅ | ❌ |

macMLX 真正独有的：**推理引擎本身就是 Swift、跑在进程内**——oMLX 的原生
app（v0.4+）壳下仍是 Python 内核，我们整个 ~50 MB DMG 里没有一行 Python。
在此之上：共享同一 Swift 核心的完整 CLI/TUI，以及用纯 Swift 拥有前沿模型
架构（DeepSeek V3.2 移植），而不是干等上游支持。

## 系统要求

macOS 14.0 (Sonoma) 或更新 · Apple Silicon（M1 或更新） · 无需 Python。

## 安装

从 [Releases](../../releases) 下载 `macMLX-vX.X.X.dmg`，挂载后把
`macMLX.app` 拖到 `/Applications`。DMG 暂未公证（[#19](../../issues/19)），
首次启动需解除 Gatekeeper：

```bash
xattr -cr /Applications/macMLX.app    # 清除隔离属性
open /Applications/macMLX.app
```

（或右键点 app → **打开** → 再 **打开**。）

## 功能亮点 (v0.2 → v0.10)

自 v0.1 MVP 起发了十八个版本，按领域。**这一节记录最新的已发布状态——
新功能先落到这里，再到下面路线图加一行。**

- **引擎与模型** —— 进程内 MLX Swift 引擎（文本 + 14 种 VLM 架构，带 OCR 模型识别；Gemma 4 与 Qwen3.5/3.6 只跑文本，模型到约 70B）；**连续批处理**（并发客户端下聚合吞吐 2.5–3.2×，只在真实并发时启用）；分层 KV prompt cache（RAM + SSD），跨 agent 轮次**最长公共前缀复用**；**投机解码**（draft 模型 + 接受率遥测）；带 LRU 淘汰的多模型池、LoRA adapter 推理、MCP server（`macmlx mcp serve`）；纯 Swift 架构移植——**DeepSeek V3.2**、**Mellum2**、Solar-Open、GLM-5.1——以外部 overlay 注册并对 Python 参考实现数值对齐（[支持分级](docs/model-support.md)）。运行在受控的 mlx-swift fork 上，携带**五条**已合入上游的修复，等某个 mlx-swift 发布版所带的 core 包含全部五条后即撤。
- **下载** —— 跨取消和退出的断点续传、实时速度/ETA、HuggingFace 镜像源、Hub commit 更新检测。
- **聊天** —— 对话侧栏（重命名、删除、回溯）、流式 Markdown、逐消息操作、按模型参数面板、可折叠 `<think>` 推理块。
- **音频** —— 基于 `mlx-audio-swift` 的进程内语音转文字与文字转语音（`POST /v1/audio/transcriptions`、`POST /v1/audio/speech`，OpenAI 形态，格式如实申报）；app 内转写是聊天输入框里可取消的附件，朗读是每条回复上的按钮；音频检查点在模型库有独立扫描。已单元测试，尚未在本机真实检查点上验证。
- **API** —— 常驻 OpenAI 兼容服务器，外加 OpenAI Responses API（`/v1/responses`，无状态——Codex 可直接接入）、Ollama（NDJSON）与 Anthropic（`/v1/messages`，支持 `thinking` 块与对话中的 system 轮次，外加 `/v1/messages/count_tokens`——Claude Code 可直接接入）兼容；**结构化输出**（`response_format` json_object / JSON-schema 子集：嵌套对象、数组、`$ref`、任意根类型、Unicode 字面量与数值范围，约束解码）；`tools` 透传与 `tool_calls` 响应；`logit_bias`、`logprobs`、`stream_options.include_usage`（usage 里带 prompt-cache 命中数）、`max_completion_tokens`、XTC 采样、按请求 LoRA adapter、KV cache 量化；`/v1/embeddings` + `/v1/rerank`（检查点是重排器时走 mlx-swift-lm 的 `MLXRerankers`——编码器交叉编码器、Qwen3 reranker、Jina v3——否则回退 bi-encoder），重排器与嵌入模型计入模型池内存预算、可按 id 卸载、可选 bearer 鉴权、模型别名 + 闲置 TTL、`reasoning_content` 分离，以及在 chat、Responses、Anthropic 三端把先前轮次回放的推理交给聊天模板（含 Codex 与 Claude Code 的工具循环）、按 ID 冷换模型、停滞看门狗、CORS + 探测端点；并发客户端批到同一模型上而不是串行排队（带 tools 的请求走单流路径，因为工具调用只在那条路径上解析）。
- **CLI** —— `pull` / `serve` / `run` 的原生 ANSI 仪表盘、与 GUI 共享 PID 协调。
- **Activity、Benchmark 与 Logs 标签页** —— 实时 **Activity 面板**，免 sudo 读取 Apple Silicon 指标（GPU 占用、内存带宽、热/内存压力、分路功耗）并给出当前推理**瓶颈**与建议，把硬件计数器与引擎自身的 prefill/decode 相位融合（外部监控工具拿不到的信号）；Benchmark 的本机 tok/s · TTFT · 峰值内存 + 社区排行榜，现在还给每轮标注 decode 瓶颈归因；Pulse 日志查看器，MLX stdout/stderr 已转入。

按 release 的完整细节见 [CHANGELOG.md](CHANGELOG.md)。

## 快速上手

**GUI** —— 启动 macMLX，Setup Wizard 选好引擎和模型目录；用内置
HuggingFace 浏览器下载模型；加载即聊。

**CLI**

```bash
macmlx pull mlx-community/Qwen3-8B-4bit     # 下载
macmlx run Qwen3-8B-4bit "你好"              # 单次提问
macmlx serve                                 # 在 :8000 启动 API
macmlx ps / stop                             # 状态 / 关闭
```

## 接入外部工具

模型加载后（或 `macmlx serve` 在跑时），OpenAI 兼容服务器常驻在
`http://localhost:8000/v1`。把任何 OpenAI 客户端（Cursor、Continue、Cline、
Open WebUI、Zed、Raycast 等）的 base URL 指过去、key 随便填即可。

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B-4bit","messages":[{"role":"user","content":"你好"}]}'
```

**Codex** 走 Responses API（`/v1/responses`）。在 `~/.codex/config.toml` 里把
macMLX 加成一个 provider，并选一个会调用工具的模型：

```toml
model = "Qwen3.6-27B-4bit"
model_provider = "macmlx"
[model_providers.macmlx]
name = "macMLX"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
# 仅在开启 bearer 鉴权时需要：Codex 会把这个环境变量的值作为 key 发送。
# env_key = "MACMLX_API_KEY"
```

macMLX 不保存任何 response，客户端每一轮都要发送完整对话（Codex 本来就这么做）；
`previous_response_id` 会被拒绝。请求逐个处理，所以 Codex 的子 agent 会排队，而不是并行运行。

**Claude Code** 走 Anthropic Messages API（`/v1/messages`）。用环境变量把它指向
macMLX，并选一个会调用工具的模型：

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_API_KEY=sk-local   # 任意值
# 开启 bearer 鉴权时，改为把 key 设给 ANTHROPIC_AUTH_TOKEN。
export ANTHROPIC_MODEL=Qwen3.6-27B-4bit
export ANTHROPIC_DEFAULT_OPUS_MODEL=Qwen3.6-27B-4bit
export ANTHROPIC_DEFAULT_SONNET_MODEL=Qwen3.6-27B-4bit
export ANTHROPIC_DEFAULT_HAIKU_MODEL=Qwen3.6-27B-4bit
# Haiku 设置的旧名字，给较旧的 Claude Code 用。
export ANTHROPIC_SMALL_FAST_MODEL=Qwen3.6-27B-4bit
claude
```

每个别名都要映射：漏掉一个，Claude Code 就会去请求某个 `claude-*` 模型 id，macMLX 会回 404。

## 推理引擎

| 引擎 | 状态 | 说明 |
|------|------|------|
| **MLX Swift**（默认） | ✅ 已发布 | Apple 的 `mlx-swift-lm`，进程内。文本 + 14 种 VLM 架构，模型到约 70B，分层 KV cache + 模型池 + LoRA。 |
| **SwiftLM**（100B+ MoE） | 🔓 可重开 | 子进程路径，sandbox 关闭后解锁（[#12](../../issues/12) / [#13](../../issues/13)）—— 尚未提上日程。 |
| **Python mlx-lm** | 🔓 可重开 | 子进程路径，换取最大模型覆盖度，代价是 PATH 里要有 `uv`。 |

所有引擎都藏在同一个 `InferenceEngine` 协议后面，GUI 永远不知道跑的是哪个。

## 架构

```
macMLX.app (SwiftUI)   macmlx (CLI)
        └──── MacMLXCore ────┘        (Swift SPM 包)
                  │
           InferenceEngine → MLXSwiftEngine（进程内）
                  │
           HummingbirdServer → http://localhost:8000/v1
                  │
           Apple Silicon (Metal / ANE)
```

数据统一在 `~/.mac-mlx/`（模型、对话、参数、日志、设置）—— 选用真实
`$HOME` 下的 dotfile，让 sandboxed 应用无需额外权限即可读写，同时对
power user 依然可见。

## 从源码构建

```bash
git clone https://github.com/magicnight/mac-mlx && cd mac-mlx
brew bundle                              # 开发工具
open macMLX/macMLX.xcodeproj             # GUI（或：xcodebuild -scheme macMLX build）
swift build --package-path macmlx-cli    # CLI
swift test  --package-path MacMLXCore    # 测试（约 3 秒）
```

## 路线图

> 这一节是项目唯一的路线图，每个 release 保持更新：某个 `0.x` 发布后，把它
> 从未来章节移到**已发布**，并同步更新上面的功能亮点。其它一切——
> `docs/superpowers/` 下的计划文档、设计稿、`.claude/features/`——要么是历史
> 记录，要么只是指回这里的指针。

- **已发布（v0.1 → v0.10.0）** —— 原生 GUI + 菜单栏 + CLI + OpenAI API（v0.1）；下载与聊天打磨（v0.2）；Benchmark、Logs、聊天历史、API 冷换、Ollama 兼容、关闭 sandbox（v0.3）；v0.5 的引擎大跃进——VLM、分层 KV cache、多模型池、LoRA、MCP server + client pool、聊天工具路由；服务端加固、embeddings + rerank、稳定性波（v0.5.1–0.5.3）；**DeepSeek V3.2 纯 Swift 移植**，`1e-4` 数值对齐；**v0.6 agent 后端**——连续批处理（并发下 2.5–3.2×）、最长公共前缀 prompt-cache 复用、结构化输出、投机解码、API 兼容包；**Track G 模型波**与按模型聊天模板覆写（v0.6.1–0.6.2）；**v0.7.0 硅指标观测**——免 sudo 的 Activity 面板、相位融合的瓶颈分类器、按轮 benchmark 归因、OCR 模型识别；**v0.8.0 分层 SSD KV cache 端到端加固**——有界、权重指纹校验、重启存活（跨会话最长前缀复用）、不阻塞；**v0.9.0**——进程内语音转文字与文字转语音（`/v1/audio/*`，加 app 内转写与朗读）、真正的 cross-encoder `/v1/rerank`、MTP drafter 检测、app 终于链接上受控 MLX fork，以及十三条上游 MLX 正确性修复；**v0.10.0**——Apple `@Generable` 发出的 `response_format` schema（嵌套对象、数组、`$ref`、任意根、Unicode 字面量、数值范围）加上 `include_usage` 与 `max_completion_tokens`、卸载真正归还内存、重排器与嵌入模型进入模型池预算、`/v1/rerank` 改跑 `MLXRerankers`、MLX fork 迁到 mlx-swift 0.32.3、音频下载留在 app 自己的缓存、DMG 按入库的 lockfile 构建。按 tag 细节见 [CHANGELOG.md](CHANGELOG.md)。
- **下个 release（在 `main` 上）** —— 暂无；见 [CHANGELOG.md](CHANGELOG.md) 的 `[Unreleased]`。
- **进行中——v0.11，本地 agent 后端。** OpenAI Responses API（`/v1/responses`，Codex 已用它取代 chat completions）与 Anthropic 的 `/v1/messages/count_tokens`；在混合注意力与滑动窗口模型（如 Qwen3.6、Mellum2）上让 prompt cache 复用每个 agent 轮次的历史；为带 MTP 头的 Qwen3.5 系检查点接上 MTP 解码（mlx-swift-lm 3.32.3 已发出其 drafter 状态），随 mlx-swift-lm 3.33 升级一起接线；以及 Xcode 27 / Foundation Models 接入，已在 Qwen3.6-27B 和 Gemma 4 E4B 上对真实服务端做过端到端验证（[配置与限制](docs/foundation-models.md)，英文；设计稿：[Foundation Models 集成设计](docs/superpowers/specs/2026-09-19-foundation-models-integration-design.md)）。
- **排队中** —— 下一个纯 Swift 架构移植，照 DeepSeek V3.2 的做法，挑 mlx-swift-lm 尚未覆盖的新架构；测量后再决定是否采用 `.balanced` prefill 分块；上游 mlx-audio-swift 的发布版做到音频下载落在传入的缓存里、且按 Hub 列表补齐已缓存快照之后切回（fork 现已基于 v0.1.5，与上游只差这些修复，含带 token 的客户端跟随 `HF_ENDPOINT` 与读完列表的每一页）；升级 swift-huggingface 1.0（破坏性版本：`HTTPClientError`、`Provider`、`Capability` 与 `Bucket.Region` 改为 struct，要求 swift-xet 1.0；还需要 swift-transformers 1.3.5 或更新（其版本范围才覆盖 1.0）以及 fork 侧匹配错误类型处的改动）。
- **更远** —— 社区 benchmark 服务；macOS 27 上把 Apple 内置模型接成 macMLX 的一个引擎；评估面向 Claude Code / Codex 的模型档位映射面板；仅当剖析需要时才做自定义 Metal kernel。
- **可重开**（sandbox 关闭后可行）—— Python / SwiftLM 子进程引擎（[#12](../../issues/12) / [#13](../../issues/13)）；签名 + 公证 DMG（[#19](../../issues/19)，需付费开发者账号）。Homebrew formula 每个 release 都会渲染并附在发布资产里，但 tap 仓库本身尚未发布（[#20](../../issues/20)）。

## 参与贡献 · 许可证

欢迎 Issue 和 PR —— 见 [CONTRIBUTING.md](CONTRIBUTING.md)。Apache 2.0
（[LICENSE](LICENSE)）。

## 鸣谢

[MLX](https://github.com/ml-explore/mlx) + [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-examples)（Apple）、[Swama](https://github.com/Trans-N-ai/swama)、[SwiftLM](https://github.com/SharpAI/SwiftLM)、[oMLX](https://github.com/jundot/omlx)、[Hummingbird](https://github.com/hummingbird-project/hummingbird)、[Sparkle](https://github.com/sparkle-project/Sparkle)、[Pulse](https://github.com/kean/Pulse)、[SwiftTUI](https://github.com/rensbreur/SwiftTUI)。完整引用：[CITATIONS.bib](CITATIONS.bib)。
