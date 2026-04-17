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

## v0.2 有什么新东西

下载和聊天体验全面升级。自 v0.1 起合入了 10 个 issue：

**下载**
- 实时分块进度条，显示下载速度 (MB/s) 与 ETA (#7)
- 下载途中可取消，残留分块自动清理 (#5)
- **跨取消和应用重启续传** —— `URLError` 里的 resume-data 会被持久化到
  `~/.mac-mlx/downloads/`，再次 Download 时从上次中断的字节开始 (#6)
- **后台 URLSession** —— 传输在 App Nap 和应用整体退出后仍继续进行；
  下次打开 macMLX 时，之前挂起的文件已经在那儿了 (#8)
- HuggingFace endpoint 可配置，支持镜像（如 `https://hf-mirror.com`），
  照顾访问 huggingface.co 慢的地区 (#21)

**聊天**
- 助手回复支持完整 Markdown 渲染，流式输出过程中也能增量显示 (#10)
- 任意消息右键菜单：Copy / Edit / Regenerate / Delete (#11)
- 对话自动保存到 `~/.mac-mlx/conversations/`，下次启动自动加载 (#9)
- **参数面板（Parameters Inspector，⌘⌥I）** —— 按模型维度保存
  temperature / top_p / max tokens / system prompt，持久化到
  `~/.mac-mlx/model-params/` (#15)

**打磨**
- 切换侧边栏标签不再中断聊天推理 (#1)
- 单实例强制 —— 重复启动会激活已存在的窗口 (#2)
- 菜单栏弹窗里加了 Quit 按钮 (#17)

完整清单：[CHANGELOG.md](CHANGELOG.md)。

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
swift test --package-path MacMLXCore   # 60 个测试，约 3 秒
```

## 路线图

### 已发布

- **v0.1.0** —— 原生 SwiftUI GUI、菜单栏、CLI（`serve` / `pull` / `run` / `list` / `ps` / `stop`）、HuggingFace 下载器、OpenAI 兼容 API、Sparkle 自动更新、内存感知的新手引导。
- **v0.2.0** —— 见上文 "v0.2 有什么新东西"。下载 + 聊天打磨；10 个 issue 关闭。

### 下一阶段（v0.3 候选）

- [#12](../../issues/12) SwiftLM 引擎（100B+ MoE）—— 待 sandbox 策略复议
- [#13](../../issues/13) Python mlx-lm 引擎 —— 待 sandbox 策略复议
- [#22](../../issues/22) Benchmark 功能（tok/s、TTFT、峰值内存）
- [#23](../../issues/23) VLM（视觉语言模型）支持
- [#16](../../issues/16) Logs 标签页（PulseUI console）
- [#20](../../issues/20) CLI 的 Homebrew tap

### 长期

- [#18](../../issues/18) 丰富的 SwiftTUI 仪表盘（上游 SwiftTUI Swift 6 兼容性阻塞中）
- [#19](../../issues/19) 签名 + 公证 DMG（等有付费 Apple Developer 账号时）

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
