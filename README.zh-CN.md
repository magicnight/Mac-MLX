# macMLX

[English](README.md) · **简体中文**

> 由 Apple MLX 驱动的原生 macOS 本地大模型推理工具。

macMLX 让 Apple Silicon 以头等原生体验跑本地 LLM 推理。无云、无遥测、
无 Electron —— 只有你的 Mac 在全速运行模型。

**面向所有人**：给新手一个精致的 SwiftUI 应用，给开发者一把趁手的 CLI。

---

## 为什么选 macMLX？

| | macMLX | LM Studio | Ollama | oMLX |
|--|--------|-----------|--------|------|
| 原生 macOS GUI | ✅ SwiftUI | ❌ Electron | ❌ | ❌ Web UI |
| 原生 MLX 推理 | ✅ | ❌ GGUF | ❌ GGUF | ✅ |
| 命令行 (CLI) | ✅ | ❌ | ✅ | ✅ |
| 断点续传 + 镜像源 | ✅ | ⚠ 部分 | ⚠ 部分 | ❌ |
| OpenAI 兼容 API | ✅ 常驻 | ✅ | ✅ | ✅ |
| 无需 Python | ✅ | ✅ | ✅ | ❌ |

## 系统要求

- macOS 14.0 (Sonoma) 或更新版本
- Apple Silicon (M1 / M2 / M3 / M4)
- 无需 Python

## 安装

从 [Releases](../../releases) 下载 `macMLX-vX.X.X.dmg`，挂载后把
`macMLX.app` 拖到 `/Applications`。

DMG **未经 Apple 公证**（暂时没有付费的 Apple Developer 账号 ——
[#19](../../issues/19)），所以首次启动 Gatekeeper 会拦截。有两种解除方式：

**方式 A —— 终端（推荐，一定有效）：**

```bash
xattr -cr /Applications/macMLX.app    # 清除 quarantine 隔离属性
open /Applications/macMLX.app         # 首次启动
```

**方式 B —— 右键打开：** 在 Finder 中右键点击 `macMLX.app` → **打开** →
在弹出的对话框里再点 **打开**。较新版 macOS 有时不会弹出这个对话框，
这种情况下改用方式 A。

想查看 Gatekeeper 对这个 app 的当前判定？

```bash
spctl --assess --verbose /Applications/macMLX.app
```

## 功能亮点（v0.2 → v0.3.7）

自 v0.1 MVP 起陆续发了十五个左右版本。挑你关心的看：

**下载**
- 可续传下载，跨取消 + 应用退出都不丢进度（后台 URLSession + resume-data 持久化）—— #5/#6/#8
- 实时速度 (MB/s) + ETA + 每文件进度条 —— #7
- Hugging Face endpoint 可配置镜像（如 `https://hf-mirror.com`）—— GUI + CLI 两端都支持 —— #21
- **HF 更新检测** —— 下载好的模型会带一个 `.macmlx-meta.json` 边车文件记录 Hub commit SHA；Models 标签页按每 24h 一次的节流向 Hub 核对，发现上游更新就显示 "Update available" 橙色徽章 —— v0.3.7

**聊天**
- **对话侧栏**：在多个历史对话间切换、重命名、删除、**回溯到指定消息**（truncate 之后的）—— v0.3.2
- 流式 Markdown 渲染，段落间距保留 —— #10（v0.3.1 修正）
- 任意消息右键：Copy / Edit / Regenerate / Delete —— #11
- 按模型的 **Parameters Inspector**（⌘⌥I）—— temperature / top_p / max tokens / system prompt 持久化到磁盘 —— #15
- Chat 工具栏模型切换器点击即加载 —— v0.3.1
- **可折叠的 `<think>` 渲染器** —— 正确展示 Qwen3 / DeepSeek-R1 / Gemma 风格的 reasoning 块 —— v0.3.6

**Benchmark** —— v0.3.0 新增标签页，测本机 tok/s、TTFT、峰值内存，带历史记录 + `Share to Community` 一键开 GitHub issue —— #22

**Logs** —— v0.3.4 新增标签页，直接读 Pulse store：搜索、按 level 过滤、实时刷新、一键清空。**MLX stdout / stderr** 在 App 启动时被 tee 进 log store（v0.3.7），不再需要 attach debugger 才能看到 `mlx-swift-lm` 的库级打印。

**API（OpenAI + Ollama 兼容）**
- 冷换模型：`/v1/chat/completions` 请求指定模型即按需加载，并发请求串行等待 —— v0.3.3
- `/x/status` 汇报真实 RSS
- **CORS 中间件 + 请求日志 + 路由别名 + 探测端点**（`GET /`、`/v1`、`/v1/health`、`/v1/status`）—— v0.3.6
- **Ollama API 兼容层** —— `GET /api/tags`、`GET /api/version`、`POST /api/chat`、`POST /api/generate`、`POST /api/show`，支持 NDJSON 流式（`stream` 字段缺省时默认流式 —— Ollama 约定）。覆盖 Zed、Immersive Translate、Open WebUI 的 Ollama provider —— v0.3.6
- **生成跨请求串行化** —— 所有 chat/completion 路径外包一个 FIFO 二元信号量，防止并发客户端把引擎打挂 —— v0.3.6

**CLI** —— 原生 ANSI 仪表盘（`macmlx pull` / `serve` / `run`），遵循 `preferredEngine` + 每模型 `ModelParameters` + HF 镜像设置。GUI 与 CLI 现在共用 `~/.mac-mlx/macmlx.pid`，拒绝在同一个 :8000 上双重绑定 —— v0.3.1 / v0.3.3 / v0.3.5 / v0.3.7

**Sandbox 关闭** —— v0.3.6 关掉 App Sandbox，`~/.mac-mlx/` 下的读写不再被重定向到 container home。和 LM Studio / Ollama / oMLX 一致。Gatekeeper 依然是用户信任层。

**稳定性 / 打磨** —— 聊天侧边栏切换不丢上下文 (#1)、单实例强制 (#2)、菜单栏 Quit (#17)、`macmlx list` segfault 修复（v0.3.1）、ConversationStore 日期精度修复（v0.3.3），以及 v0.3.6 的 13 个用户上报 bug + 一打 post-QA hot patch

按 tag 的完整变更：[CHANGELOG.md](CHANGELOG.md)。

## 快速上手

### GUI
1. 启动 macMLX，首次启动的 Setup Wizard 会把你指引到 `~/.mac-mlx/models` 并选中 MLX Swift 引擎
2. 用内置的 HuggingFace 浏览器下载模型（支持续传、走镜像）
3. 加载模型，开始聊天

### CLI

```bash
macmlx pull mlx-community/Qwen3-8B-4bit     # 下载
macmlx list                                  # 本地模型列表
macmlx run Qwen3-8B-4bit "你好"              # 单次提问
macmlx run Qwen3-8B-4bit                     # 进入交互模式
macmlx serve                                 # 在 :8000 启动 API
macmlx ps                                    # 检查 serve 是否在跑
macmlx stop                                  # 优雅的 SIGTERM
```

## 接入外部工具

当模型加载后（或 `macmlx serve` 在跑时），OpenAI 兼容服务器会常驻在
`http://localhost:8000/v1`。

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B-4bit","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

任何 OpenAI 兼容客户端都能直接用 —— 把 base URL 指向
`http://localhost:8000/v1`，API key 随便填：

- Cursor / Continue / Cline：在设置里填自定义 base URL
- Open WebUI：添加一个 OpenAI provider
- Raycast、Zed 等等：同样的套路

## 推理引擎

| 引擎 | 状态 | 说明 |
|------|------|------|
| **MLX Swift**（默认） | ✅ 已发布 | Apple 的 `mlx-swift-lm`，进程内运行。在 64GB+ 的 Mac 上支持到约 70B 的模型。v0.4.0 起支持分层 KV prompt cache + 多模型池。 |
| **SwiftLM**（100B+ MoE） | 🔓 可重开 | 子进程路径之前被 App Sandbox 拦住，v0.3.6 关掉 sandbox 后 [#12](../../issues/12) / [#13](../../issues/13) 可重新考虑 —— 候选 v0.5/v0.6，尚未提上日程。填补 [mlx-swift-lm#219](https://github.com/ml-explore/mlx-swift-lm/issues/219) 的 MoE 空白。 |
| **Python mlx-lm** | 🔓 可重开 | 同一套子进程路径。代价是 PATH 里需要 `uv`，换取 mlx-community 的 Python-only checkpoint 最大覆盖度。 |

目前的 Settings → Engine 里，非默认引擎会显示 Install Guide 链接；
选中它们时会优雅降级到 "engine not available" 状态。

## 架构

```
macMLX.app (SwiftUI)        macmlx (CLI)
            │                    │
            └─────── MacMLXCore ─┘    (Swift SPM 包)
                        │
               InferenceEngine
                        │
                  MLXSwiftEngine    (进程内, mlx-swift-lm 3.31.x)
                        │
                  HummingbirdServer  → http://localhost:8000/v1
                        │
                Apple Silicon (Metal / ANE)
```

数据目录统一在 `~/.mac-mlx/`：

```
~/.mac-mlx/
├── models/              # 权重（默认路径，可在 Settings 更改）
├── conversations/       # 聊天历史 JSON
├── model-params/        # 每个模型的参数覆盖
├── downloads/           # 中断下载的 resume-data
├── logs/                # Pulse 日志
├── settings.json        # 用户偏好
└── macmlx.pid           # CLI 守护进程协调
```

刻意选用 `$HOME` 下的 dotfile 路径：macOS App Sandbox 对 dotfile 有豁免
规则，让 sandboxed 应用无需 `user-selected.read-write` 权限或安全作用
域书签也能读写这里，同时对 power user 在 Finder 里依然可见。

## 从源码构建

```bash
git clone https://github.com/magicnight/mac-mlx
cd mac-mlx
brew bundle                            # 开发工具

# GUI 应用
open macMLX/macMLX.xcodeproj           # 或：xcodebuild -scheme macMLX build

# CLI
swift build --package-path macmlx-cli

# 核心包 + 测试
swift test --package-path MacMLXCore   # 约 3 秒
```

## 路线图

### 已发布

- **v0.1.0** —— 原生 SwiftUI GUI、菜单栏、CLI（`serve` / `pull` / `run` / `list` / `ps` / `stop`）、HuggingFace 下载器、OpenAI 兼容 API、Sparkle 自动更新、内存感知的新手引导。
- **v0.2.0** —— 下载 + 聊天打磨（10 个 issue）：续传下载、HF 镜像、Markdown 渲染、消息编辑/重生成、参数面板。
- **v0.3.0 → v0.3.5** —— Benchmark 功能、跨切面缺陷修复、UX 小修、聊天历史侧栏、API 冷换模型、Logs 标签页、原生 ANSI CLI 仪表盘。
- **v0.3.6** —— 13 个用户上报 bug + 一打 post-QA hot patch：可折叠 `<think>` 渲染器、关闭 App Sandbox、CORS + 请求日志 + 路由别名、Ollama API 兼容层 + NDJSON 流式、GUI/CLI 通过 `LoadHook` 统一状态、FIFO 生成信号量、聊天渲染修复、侧栏重建。
- **v0.3.7** —— 维护 release：CI 升级到 Node.js 24（`actions/checkout@v5` / `actions/cache@v5`）、MLX stdout / stderr 转进 Logs 标签页、通过 `.macmlx-meta.json` 边车文件做 HF 更新检测、GUI 与 CLI 共用 `~/.mac-mlx/macmlx.pid`。

按 tag 的细节见 `CHANGELOG.md`。

### 进行中（v0.4.0 —— 对标 oMLX 的引擎升级）

从原先 "VLM 优先" 的计划转向：对比 [oMLX](https://github.com/jundot/omlx)（10.6k★）之后，更高 leverage 的投入是先补上推理引擎的差距。VLM 推后到 v0.4.1。三个独立子项目，同一个 release：

- **分层 KV cache（hot RAM + cold SSD）** —— 已合并到 `main`（PR #26）。同一个模型后续轮次对话会复用 KV cache：新提示是旧提示的延长时共享前缀就跳过 prefill。热层 = 内存里的 LRU 字典，冷层 = `~/.mac-mlx/kv-cache/` 下 16-way 分片的 safetensors，走 mlx-swift-lm 的 `savePromptCache` / `loadPromptCache` 往返。Settings → "KV Cache" 有热/冷预算 stepper + Clear All 按钮。对"编程助手"式工作流（Claude Code / Cursor / Zed 每轮都重发整段历史）显著降低 TTFT。
- **多模型池 + 自动换出** —— 在 PR #27 里。`ModelPool` actor 持有 `[String: InferenceEngine]`，按用户可配的常驻内存预算上限（Settings → Model Pool；默认 = 总内存 50%）约束。超预算时非 pin 模型按 LRU 自动 evict。在 Models 标签页里点某行的橙色 pin 图标就可以强制常驻。pin 模型之间冷换不再需要重新读权重。
- **MCP server MVP** —— 下一步。新 CLI 子命令 `macmlx mcp serve`，通过 stdio 走 [`modelcontextprotocol/swift-sdk`](https://github.com/modelcontextprotocol/swift-sdk) v0.11.x，暴露 `list_models` 和 `chat` 两个工具。在 Claude Desktop / Cursor 的 `mcpServers` 配置里加一行就能让本地 MLX 推理通过它们的工具生态被调用。

完整计划：[`docs/roadmap-post-v0.3.6.md`](docs/roadmap-post-v0.3.6.md)。

### 下个 minor 版本（v0.4.1 —— VLM）

原先 v0.4 的 scope 保持不变，只是平移一位：

- [#23](../../issues/23) VLM（视觉语言模型）支持，通过 `MLXVLM`（已经在依赖里）。16 种架构：Qwen2.5-VL / Qwen3-VL / Gemma-3 / SmolVLM/2 / Paligemma / Pixtral / Idefics3 / FastVLM / LFM2-VL / glm_ocr / mistral3。图像选择器（NSOpenPanel + 拖放 + 粘贴）、`HummingbirdServer` 解析 OpenAI 多模态 `content` 数组、图像写入 `~/.mac-mlx/conversations/<uuid>/images/`。

### 更远（v0.5 起）

- **v0.5** —— 连续批处理（阻塞在上游 `mlx-swift-lm` 什么时候把 `BatchGenerator` + `BatchKVCache` 从 Python 移植过来，参考 Python mlx-lm PR [#941](https://github.com/ml-explore/mlx-lm/pull/941) / [#1101](https://github.com/ml-explore/mlx-lm/pull/1101)）、LoRA adapter 加载（HF 现成 adapter 直接用，不含训练 UI）、MCP *client*（在 macMLX 里配置外部 MCP server，让聊天模型通过它们做工具调用）。
- **v0.6** —— 语音 I/O，通过 [`DePasqualeOrg/mlx-swift-audio`](https://github.com/DePasqualeOrg/mlx-swift-audio)（替换原先 WhisperKit 方案）。MLX 原生 STT（Whisper、中文强的 Fun-ASR）+ TTS（流式 Marvis、可复刻声音的 Chatterbox、CosyVoice 2）。故意不纳入 Kokoro —— 它间接依赖 GPL-3 的 espeak-ng。
- **v0.7** —— Community Benchmarks 服务。可选的 `POST /v1/benchmarks` 端点接收匿名化的 `BenchmarkResult` + `HardwareInfo`，按芯片 × 模型 × 量化 × macOS 版本聚合到网站和 App 内的公开排行榜。

### 关闭 sandbox 后可以重开（v0.3.6）

v0.3.6 关掉了 App Sandbox，之前标 "not planned" 的项目重新变得可行。目前没一项提上日程：

- [#12](../../issues/12) Python `mlx-lm` 引擎走子进程 —— 最大覆盖度，代价是需要 PATH 里有 `uv`，first-token 略慢。
- [#13](../../issues/13) SwiftLM 二进制引擎走子进程 —— 填上 `mlx-swift-lm` 搞不定的 100B+ MoE（Gemma 4 MoE / Llama 4 MoE / DeepSeek-V3）。
- [#20](../../issues/20) CLI 的 Homebrew tap —— 等 CLI tarball 作为 release asset 一起打后就能推。

### 还在 defer / 被阻塞

- [#19](../../issues/19) 签名 + 公证 DMG —— 需要付费 Apple Developer 账号。

## 参与贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。Issue 和 PR 都欢迎。

## 许可证

Apache 2.0 —— 见 [LICENSE](LICENSE)

## 鸣谢

- Apple 的 [MLX](https://github.com/ml-explore/mlx) 与 [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-examples)
- [Swama](https://github.com/Trans-N-ai/swama) —— Swift 推理架构的灵感来源
- [SwiftLM](https://github.com/SharpAI/SwiftLM) —— 100B+ MoE 引擎（未来整合）
- [oMLX](https://github.com/jundot/omlx) —— 功能深度的参考
- [Hummingbird](https://github.com/hummingbird-project/hummingbird) —— Swift HTTP 服务器
- [Sparkle](https://github.com/sparkle-project/Sparkle) —— 自动更新框架
- [Pulse](https://github.com/kean/Pulse) —— 日志框架
- [SwiftTUI](https://github.com/rensbreur/SwiftTUI) —— TUI 框架

完整 BibTeX 引用：[CITATIONS.bib](CITATIONS.bib)
