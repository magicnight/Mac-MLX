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

macOS 14.0 (Sonoma) 或更新 · Apple Silicon (M1–M4) · 无需 Python。

## 安装

从 [Releases](../../releases) 下载 `macMLX-vX.X.X.dmg`，挂载后把
`macMLX.app` 拖到 `/Applications`。DMG 暂未公证（[#19](../../issues/19)），
首次启动需解除 Gatekeeper：

```bash
xattr -cr /Applications/macMLX.app    # 清除隔离属性
open /Applications/macMLX.app
```

（或右键点 app → **打开** → 再 **打开**。）

## 功能亮点 (v0.2 → v0.5.3)

自 v0.1 MVP 起发了十六个以上版本，按领域。**这一节记录最新的已发布状态——
新功能先落到这里，再到下面路线图加一行。**

- **引擎与模型** —— 进程内 MLX Swift 引擎（文本 + 16 种 VLM 架构，模型到约 70B）；分层 KV prompt cache（RAM + SSD）、带 LRU 淘汰的多模型池、LoRA adapter 推理、MCP server（`macmlx mcp serve`）；纯 Swift **DeepSeek V3.2** 架构（DSA 稀疏注意力 + absorbed MLA + MoE），零 fork overlay 注册，对 Python 参考逐组件数值对齐。
- **下载** —— 跨取消和退出的断点续传、实时速度/ETA、HuggingFace 镜像源、Hub commit 更新检测。
- **聊天** —— 对话侧栏（重命名、删除、回溯）、流式 Markdown、逐消息操作、按模型参数面板、可折叠 `<think>` 推理块。
- **API** —— 常驻 OpenAI 兼容服务器，外加 Ollama（NDJSON）与 Anthropic（`/v1/messages`）兼容、`/v1/embeddings` + `/v1/rerank`、可选 bearer 鉴权、模型别名 + 闲置 TTL、`reasoning_content` 分离、按 ID 冷换模型、停滞看门狗、CORS + 探测端点、生成跨客户端串行化。
- **CLI** —— `pull` / `serve` / `run` 的原生 ANSI 仪表盘、与 GUI 共享 PID 协调。
- **Benchmark 与 Logs 标签页** —— 本机 tok/s · TTFT · 峰值内存 + 社区排行榜；Pulse 日志查看器，MLX stdout/stderr 已转入。

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

## 推理引擎

| 引擎 | 状态 | 说明 |
|------|------|------|
| **MLX Swift**（默认） | ✅ 已发布 | Apple 的 `mlx-swift-lm`，进程内。文本 + 16 种 VLM 架构，模型到约 70B，分层 KV cache + 模型池 + LoRA。 |
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

> 每个 release 保持更新：某个 `0.x` 发布后，把它从未来章节移到**已发布**，
> 并同步更新上面的功能亮点。

- **已发布（v0.1 → v0.5）** —— 原生 GUI + 菜单栏 + CLI + OpenAI API（v0.1）；下载与聊天打磨（v0.2）；Benchmark、Logs、聊天历史、API 冷换、Ollama 兼容、关闭 sandbox（v0.3）；以及 v0.5 的引擎大跃进——VLM、分层 KV cache、多模型池、LoRA、MCP server。按 tag 细节见 [CHANGELOG.md](CHANGELOG.md)。
- **下个 release（在 `main` 上）** —— 服务端加固：api-key 鉴权、Anthropic `/v1/messages`、别名 + 闲置 TTL、模板 kwargs（v0.5.1）；embeddings + rerank 端点（v0.5.2）；server/pool 稳定性波——换模与生成原子化、不泄漏的生成锁、停滞看门狗、模型池 pin + 真取消（v0.5.3）；MCP client pool；`reasoning_content` 分离（[#30](../../issues/30)）；以及 **DeepSeek V3.2 纯 Swift 移植**——DSA 稀疏注意力 + absorbed MLA + MoE 以零 fork overlay 注册进 mlx-swift-lm 工厂，逐组件对 Python 参考 `1e-4` 数值对齐。在 Ollama 和 LM Studio 也上了 MLX 后端的当下，这是 macMLX 的差异化。
- **进行中** —— 接进聊天的 MCP 工具路由；DeepSeek 后续（真权重 smoke，然后 V4 增量）；真 cross-encoder 重排。（debug 轮的 server/pool 加固 backlog 已全部落地——PRs #55-#57。）
- **下一版（v0.6）—— agent 后端** —— 连续批处理（基于上游批缓存原语自研编排器）、跨轮最长公共前缀 prompt-cache 复用、结构化输出（JSON Schema 约束解码）、投机解码接线（draft 模型 + MTP）、API 兼容包（`logit_bias` / `logprobs` / 每请求 adapter / server `tools` 透传）、GUI 升级（已有 HF 缓存发现、coding-agent Integrations 屏、模型卡打磨），以及纯 Swift 模型移植流水线（Llama 4、Command R7B、Kimi、MiniCPM3……）。
- **更远（v0.7+）** —— 语音 I/O（MLX 原生 STT/TTS）；社区 benchmark 服务；若性能剖析需要，为我们的 DeepSeek DSA 路径做自定义 Metal kernel。
- **可重开**（sandbox 关闭后可行）—— Python / SwiftLM 子进程引擎（[#12](../../issues/12) / [#13](../../issues/13)）、Homebrew tap（[#20](../../issues/20)）、签名 + 公证 DMG（[#19](../../issues/19)）。

## 参与贡献 · 许可证

欢迎 Issue 和 PR —— 见 [CONTRIBUTING.md](CONTRIBUTING.md)。Apache 2.0
（[LICENSE](LICENSE)）。

## 鸣谢

[MLX](https://github.com/ml-explore/mlx) + [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-examples)（Apple）、[Swama](https://github.com/Trans-N-ai/swama)、[SwiftLM](https://github.com/SharpAI/SwiftLM)、[oMLX](https://github.com/jundot/omlx)、[Hummingbird](https://github.com/hummingbird-project/hummingbird)、[Sparkle](https://github.com/sparkle-project/Sparkle)、[Pulse](https://github.com/kean/Pulse)、[SwiftTUI](https://github.com/rensbreur/SwiftTUI)。完整引用：[CITATIONS.bib](CITATIONS.bib)。
