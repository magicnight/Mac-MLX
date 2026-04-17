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

## 功能亮点（v0.2 → v0.3.5）

自 v0.1 MVP 起陆续发了十多个版本。挑你关心的看：

**下载**
- 可续传下载，跨取消 + 应用退出都不丢进度（后台 URLSession + resume-data 持久化）—— #5/#6/#8
- 实时速度 (MB/s) + ETA + 每文件进度条 —— #7
- Hugging Face endpoint 可配置镜像（如 `https://hf-mirror.com`）—— GUI + CLI 两端都支持 —— #21

**聊天**
- **对话侧栏**：在多个历史对话间切换、重命名、删除、**回溯到指定消息**（truncate 之后的）—— v0.3.2
- 流式 Markdown 渲染，段落间距保留 —— #10（v0.3.1 修正）
- 任意消息右键：Copy / Edit / Regenerate / Delete —— #11
- 按模型的 **Parameters Inspector**（⌘⌥I）—— temperature / top_p / max tokens / system prompt 持久化到磁盘 —— #15
- Chat 工具栏模型切换器点击即加载 —— v0.3.1

**Benchmark** —— v0.3.0 新增标签页，测本机 tok/s、TTFT、峰值内存，带历史记录 + `Share to Community` 一键开 GitHub issue —— #22

**Logs** —— v0.3.4 新增标签页，直接读 Pulse store：搜索、按 level 过滤、实时刷新、一键清空

**API（OpenAI 兼容）**
- 冷换模型：`/v1/chat/completions` 请求指定模型即按需加载，并发请求串行等待 —— v0.3.3
- `/x/status` 汇报真实 RSS

**CLI** —— 原生 ANSI 仪表盘（`macmlx pull` / `serve` / `run`），遵循 `preferredEngine` + 每模型 `ModelParameters` + HF 镜像设置 —— v0.3.1 / v0.3.3 / v0.3.5

**稳定性 / 打磨** —— 聊天侧边栏切换不丢上下文 (#1)、单实例强制 (#2)、菜单栏 Quit (#17)、`macmlx list` segfault 修复（v0.3.1）、ConversationStore 日期精度修复（v0.3.3），以及 v0.3.0 一轮独立代码评审扫了一大批坑

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
| **MLX Swift**（默认） | ✅ 已发布 | Apple 的 `mlx-swift-lm`，进程内运行。在 64GB+ 的 Mac 上支持到约 70B 的模型。 |
| **SwiftLM**（100B+ MoE） | 🕒 延后到 v0.3 | 启动子进程被 App Sandbox 策略阻止；等有明确用户诉求时再复议（[#12](../../issues/12)）。 |
| **Python mlx-lm** | 🕒 延后到 v0.3 | 同样的 sandbox 阻塞（[#13](../../issues/13)）。 |

目前的 Settings → Engine 里，被延后的引擎会显示 Install Guide 链接；
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
swift test --package-path MacMLXCore   # 90 个测试，约 3 秒
```

## 路线图

### 已发布

- **v0.1.0** —— 原生 SwiftUI GUI、菜单栏、CLI（`serve` / `pull` / `run` / `list` / `ps` / `stop`）、HuggingFace 下载器、OpenAI 兼容 API、Sparkle 自动更新、内存感知的新手引导。
- **v0.2.0** —— 下载 + 聊天打磨（10 个 issue）：续传下载、HF 镜像、Markdown 渲染、消息编辑/重生成、参数面板。
- **v0.3.x** —— 六个 patch release：Benchmark 功能、跨切面缺陷修复、UX 小修、聊天历史侧栏、API 冷换模型、Logs 标签页、原生 ANSI CLI 仪表盘。按 tag 的细节见 `CHANGELOG.md`。

### 下一版本（v0.3.6 —— 维护补丁）

- `macmlx --version` 自动同步 tag
- `macmlx search <query>` 命令（默认查 `mlx-community`）
- 二进制精简：`strip -S` + Swift stdlib 动态链接
- CLI 加 `--log-level` + `--log-stderr` 开关让 Pulse 日志在终端里可见

### 下个 minor 版本（v0.4.0）

- [#23](../../issues/23) VLM（视觉语言模型）支持 —— `MLXVLM` 已经在依赖里，16 种架构（Qwen2.5-VL / SmolVLM / Gemma-3 / Paligemma 等）。完整计划见 [`.omc/plans/v0.4-vlm-plan.md`](.omc/plans/v0.4-vlm-plan.md)。

### 更远（v0.5 起）

- **v0.5** —— LoRA adapter 加载（HF 上现成 adapter 直接用，不含训练）+ 对话/数据集导出
- **v0.6** —— 语音 I/O：WhisperKit 做 ASR（聊天麦克风输入）+ AVSpeechSynthesizer 做 TTS（助手回复朗读）
- [#20](../../issues/20) CLI 的 Homebrew tap（v0.3.6–v0.4 期间等 CLI tarball 作为 release asset 一起打后再推）

### 已 defer / 被阻塞

- [#19](../../issues/19) 签名 + 公证 DMG —— 需要付费 Apple Developer 账号
- Swift 原生 MLX Whisper —— 上游 `mlx-swift-lm` 还没提供音频模型；当下用 WhisperKit（Core ML）覆盖体验
- [#12](../../issues/12) / [#13](../../issues/13) 子进程引擎（SwiftLM、Python mlx-lm）—— 关闭为 *not planned*，因为 App Sandbox 禁止 spawn 外部二进制。若 sandbox 策略未来调整或出现 Swift 原生 100B+ MoE 推理方案，可重新打开。

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
