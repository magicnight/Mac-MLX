const repository = "https://github.com/magicnight/mac-mlx";
const release = `${repository}/releases/tag/v0.5.3`;
const tagged = (path) => `${repository}/blob/v0.5.3/${path}`;
const main = (path) => `${repository}/blob/main/${path}`;
const verified = "2026-07-10";

function fact(id, status, sinceVersion, pageIds, sources, en, zh) {
  return Object.freeze({
    id,
    status,
    sinceVersion,
    lastVerified: verified,
    pageIds: Object.freeze(pageIds),
    sourceUrls: Object.freeze(sources),
    en: Object.freeze(en),
    "zh-Hans": Object.freeze(zh),
  });
}

const changelog = tagged("CHANGELOG.md");
const engine = tagged("MacMLXCore/Sources/MacMLXCore/Engine/MLXSwiftEngine.swift");
const server = tagged("MacMLXCore/Sources/MacMLXCore/Server/HummingbirdServer.swift");
const roadmap = main("docs/superpowers/specs/2026-07-10-engine-scroll-story-design.md");

export const facts = Object.freeze([
  fact("platform-installation", "released", "0.1.0", [], [release, tagged("README.md")],
    { title: "Apple Silicon macOS installation", summary: "macMLX supports Apple Silicon Macs running macOS 14 or later.", detail: "Use the current project installation and Gatekeeper guidance; do not disable system-wide security protections solely to open the app." },
    { title: "Apple 芯片 macOS 安装", summary: "macMLX 支持运行 macOS 14 或更高版本的 Apple 芯片 Mac。", detail: "请使用项目当前安装与 Gatekeeper 指南；不要仅为打开应用而关闭系统级安全保护。" }),
  fact("swift-in-process", "released", "0.1.0", ["architecture"], [release, engine],
    { title: "Swift in-process inference", summary: "The default engine loads and runs MLX models inside the Swift process.", detail: "Model loading, generation, caching, and serving use Apple MLX through MacMLXCore; the default inference path does not require a Python runtime." },
    { title: "Swift 进程内推理", summary: "默认引擎在 Swift 进程内加载并运行 MLX 模型。", detail: "模型加载、生成、缓存与服务通过 MacMLXCore 使用 Apple MLX；默认推理路径不需要 Python 运行时。" }),
  fact("unified-memory", "released", "0.1.0", ["architecture", "models", "choosing-a-model", "vision-language-models"], ["https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html", engine],
    { title: "Apple Silicon unified memory", summary: "MLX arrays use the Mac's shared CPU/GPU memory system.", detail: "Unified memory reduces explicit transfers between CPU orchestration and integrated-GPU compute, but model weights, activations, and KV cache still consume finite physical memory." },
    { title: "Apple 芯片统一内存", summary: "MLX 数组使用 Mac 的 CPU/GPU 共享内存系统。", detail: "统一内存减少 CPU 编排与集成 GPU 计算之间的显式传输，但模型权重、激活值和 KV 缓存仍会占用有限的物理内存。" }),
  fact("shared-core", "released", "0.1.0", ["architecture"], [release, tagged("MacMLXCore/Sources/MacMLXCore/Server/HummingbirdServer.swift")],
    { title: "Shared code, process-local engines", summary: "The app and CLI both import MacMLXCore, which owns inference and the server.", detail: "The products share implementation and behavior. When the app and CLI run as separate processes, they do not share one in-memory engine instance." },
    { title: "共享代码，进程内各自运行", summary: "应用与 CLI 都导入 MacMLXCore，推理和服务由核心负责。", detail: "两个产品共享实现与行为；当应用和 CLI 分别运行在不同进程时，它们不会共享同一个内存中引擎实例。" }),
  fact("no-python-default", "released", "0.1.0", ["architecture"], [release, engine],
    { title: "No Python on the default path", summary: "The released default engine is Swift-native and needs no Python runtime.", detail: "Optional compatibility engines may use subprocesses and other runtimes. This is not a claim that Python is absent everywhere in the project." },
    { title: "默认路径无需 Python", summary: "已发布的默认引擎为 Swift 原生，不需要 Python 运行时。", detail: "可选兼容引擎可能使用子进程与其他运行时，因此这并不表示项目任何地方都没有 Python。" }),
  fact("openai-compat", "released", "0.5.3", ["api-compatibility"], [release, changelog, server],
    { title: "OpenAI endpoint compatibility", summary: "Chat, legacy completions, model listing, and embeddings use compatible request and response shapes.", detail: "Compatibility is endpoint-specific. macMLX model load and unload routes under /x/models are project extensions, not OpenAI-compatible model management." },
    { title: "OpenAI 端点兼容", summary: "聊天、传统补全、模型列表与嵌入使用兼容的请求和响应结构。", detail: "兼容范围按端点界定。/x/models 下的模型加载与卸载是 macMLX 扩展，不属于 OpenAI 兼容的模型管理。" }),
  fact("anthropic-messages", "released", "0.5.3", ["api-compatibility"], [release, changelog, server],
    { title: "Anthropic Messages compatibility", summary: "POST /v1/messages, including streaming, is available in v0.5.3.", detail: "This is Messages API compatibility only, not compatibility with the full Anthropic API." },
    { title: "Anthropic Messages 兼容", summary: "v0.5.3 提供 POST /v1/messages，并支持流式响应。", detail: "这里只兼容 Messages API，不代表兼容完整 Anthropic API。" }),
  fact("ollama-selected", "released", "0.3.7", ["api-compatibility"], [release, changelog, server],
    { title: "Selected Ollama endpoints", summary: "macMLX supports /api/version, /api/tags, /api/show, /api/chat, and /api/generate.", detail: "The compatibility layer has shipped since v0.3.7. It is a selected endpoint set, not a drop-in replacement for every Ollama API." },
    { title: "部分 Ollama 端点", summary: "macMLX 支持 /api/version、/api/tags、/api/show、/api/chat 与 /api/generate。", detail: "该兼容层自 v0.3.7 起提供；它只覆盖选定端点，不是完整 Ollama API 的无差别替代。" }),
  fact("mcp-server", "released", "0.5.0", ["api-compatibility"], [release, changelog],
    { title: "MCP server", summary: "The CLI can expose local inference to MCP clients.", detail: "The MCP server shipped in v0.5.0 and is separate from chat-side routing to external tools." },
    { title: "MCP 服务端", summary: "CLI 可向 MCP 客户端提供本地推理能力。", detail: "MCP 服务端在 v0.5.0 发布，与聊天侧调用外部工具的路由功能不同。" }),
  fact("mcp-client-pool", "released", "0.5.3", ["api-compatibility"], [release, changelog, tagged("MacMLXCore/Sources/MacMLXCore/MCP/MCPClientPool.swift")],
    { title: "MCP client pool", summary: "v0.5.3 includes managed MCP client connections.", detail: "The pool manages external MCP processes and connections; integrated chat-side tool routing is still development work." },
    { title: "MCP 客户端池", summary: "v0.5.3 包含受管理的 MCP 客户端连接。", detail: "客户端池负责管理外部 MCP 进程与连接；聊天侧集成工具路由仍属于开发工作。" }),
  fact("integrated-tool-routing", "development", "post-0.5.3", ["api-compatibility"], [main("README.md")],
    { title: "Integrated chat tool routing", summary: "Chat-side routing through configured MCP tools is being developed after v0.5.3.", detail: "It must not be confused with the released MCP server or client-pool infrastructure." },
    { title: "聊天集成工具路由", summary: "通过已配置 MCP 工具进行聊天侧路由，是 v0.5.3 之后的开发工作。", detail: "它不能与已发布的 MCP 服务端或客户端池基础设施混为一谈。" }),
  fact("embeddings", "released", "0.5.3", ["api-compatibility"], [release, changelog, tagged("MacMLXCore/Sources/MacMLXCore/Engine/EmbeddingEngine.swift")],
    { title: "Local embeddings", summary: "POST /v1/embeddings shipped in v0.5.3.", detail: "Encoder-family model detection exists, while using an unsuitable chat model can still produce vectors without semantic guarantees." },
    { title: "本地嵌入", summary: "POST /v1/embeddings 已在 v0.5.3 发布。", detail: "系统可以检测编码器模型家族，但若使用不合适的聊天模型，仍可能生成缺乏语义保证的向量。" }),
  fact("rerank-bi-encoder", "released", "0.5.3", ["api-compatibility"], [release, changelog, tagged("MacMLXCore/Sources/MacMLXCore/Engine/RerankScoring.swift")],
    { title: "Bi-encoder rerank MVP", summary: "POST /v1/rerank scores independently embedded texts with cosine similarity.", detail: "This released MVP is not a cross-encoder reranker." },
    { title: "双编码器重排 MVP", summary: "POST /v1/rerank 对独立嵌入的文本计算余弦相似度。", detail: "这一已发布 MVP 不是交叉编码器重排器。" }),
  fact("tiered-cache", "released", "0.5.0", ["architecture", "choosing-a-model"], [release, changelog, tagged("MacMLXCore/Sources/MacMLXCore/PromptCache/PromptCacheStore.swift")],
    { title: "Exact-prefix RAM and SSD cache", summary: "A hot RAM tier and content-addressed SSD cold tier support promotion and demotion.", detail: "The released v0.5.0 cache reuses exact full prefixes. It does not provide released block sharing or paged KV allocation." },
    { title: "精确前缀 RAM 与 SSD 缓存", summary: "内存热层与内容寻址 SSD 冷层支持提升和降级。", detail: "v0.5.0 发布的缓存复用完整精确前缀，不包含已发布的块共享或分页 KV 分配。" }),
  fact("trie-lcp", "development", "post-0.5.3", ["architecture"], [main("MacMLXCore/Sources/MacMLXCore/PromptCache/PromptTrie.swift"), main("MacMLXCore/Sources/MacMLXCore/Engine/MLXSwiftEngine.swift")],
    { title: "Trie longest-prefix reuse", summary: "Post-tag main can reuse the longest compatible cached token prefix.", detail: "This work is development-only and must not be attributed to the v0.5.3 release." },
    { title: "Trie 最长前缀复用", summary: "标签之后的 main 分支可复用最长兼容缓存词元前缀。", detail: "该工作仅属于开发状态，不能归入 v0.5.3 发布版。" }),
  fact("model-pool", "released", "0.5.0", ["architecture", "models", "choosing-a-model"], [release, changelog, tagged("MacMLXCore/Sources/MacMLXCore/ModelPool/ModelPool.swift")],
    { title: "Bounded model pool", summary: "Budgets, LRU eviction, pinning, cold swap, idle TTL, and probes bound multi-model use.", detail: "The pool shipped in v0.5.0 and was hardened in v0.5.3. It is not a unified adaptive controller." },
    { title: "有界模型池", summary: "预算、LRU 淘汰、固定、冷切换、空闲 TTL 与探针共同约束多模型使用。", detail: "模型池在 v0.5.0 发布并于 v0.5.3 加固；它不是统一自适应控制器。" }),
  fact("lora", "released", "0.5.0", ["models"], [release, changelog],
    { title: "Supported LoRA adapters", summary: "The native engine can apply supported LoRA adapters.", detail: "Adapter compatibility depends on the base architecture and weights; universal LoRA compatibility is not claimed." },
    { title: "受支持的 LoRA 适配器", summary: "原生引擎可应用受支持的 LoRA 适配器。", detail: "适配器兼容性取决于基础架构与权重，不宣称通用 LoRA 兼容。" }),
  fact("vlm-14-families", "released", "0.5.3", ["models", "vision-language-models"], [release, tagged("MacMLXCore/Sources/MacMLXCore/Managers/ModelLibraryManager.swift")],
    { title: "Fourteen detected VLM families", summary: "The model library detects 14 vision-language model_type families.", detail: "This is an evidence-backed family count, not a guarantee that every checkpoint or processor variant will load." },
    { title: "检测 14 个 VLM 家族", summary: "模型库可检测 14 个视觉语言 model_type 家族。", detail: "这是有证据支持的家族数量，不保证每个检查点或处理器变体都能加载。" }),
  fact("deepseek-v32", "released", "0.5.3", ["models"], [release, changelog, tagged("MacMLXCore/Sources/MacMLXCore/Models/DeepseekV32.swift")],
    { title: "DeepSeek V3.2 Swift overlay", summary: "v0.5.3 includes pure-Swift component parity for the DeepSeek V3.2 architecture.", detail: "A real-checkpoint smoke test remains pending and FP8 dequantization is absent, so this is not an end-to-end or universal MoE claim." },
    { title: "DeepSeek V3.2 Swift 覆盖层", summary: "v0.5.3 包含 DeepSeek V3.2 架构的纯 Swift 组件对齐实现。", detail: "真实检查点冒烟测试仍待完成，且缺少 FP8 反量化，因此不构成端到端或通用 MoE 声明。" }),
  fact("continuous-batching", "development", "post-0.5.3", ["architecture", "api-compatibility", "vision-language-models"], [main("MacMLXCore/Sources/MacMLXCore/Batching/BatchScheduler.swift"), main("MacMLXCore/Sources/MacMLXCore/Server/BatchRoutingPolicy.swift")],
    { title: "Limited continuous batching", summary: "Post-tag development batches eligible dense text models for OpenAI chat and completions.", detail: "VLM, speculative, Ollama, Anthropic, and embeddings paths remain serial." },
    { title: "受限连续批处理", summary: "标签之后的开发版本为符合条件的稠密文本模型批处理 OpenAI 聊天与补全。", detail: "VLM、推测解码、Ollama、Anthropic 与嵌入路径仍为串行。" }),
  fact("fixed-prefill-throttle", "development", "post-0.5.3", ["architecture"], [main("MacMLXCore/Sources/MacMLXCore/Batching/BatchScheduler.swift")],
    { title: "Fixed prefill admission throttle", summary: "A fixed prefillBatchSize bounds rows admitted per scheduler step.", detail: "This development throttle is fixed configuration, not an adaptive memory controller." },
    { title: "固定预填充准入节流", summary: "固定 prefillBatchSize 限制每个调度步骤接纳的行数。", detail: "这一开发中节流是固定配置，不是自适应内存控制器。" }),
  fact("paged-kv", "planned", "future", ["architecture"], [roadmap],
    { title: "Paged KV, block sharing, and CoW", summary: "Paged allocation, shared blocks, and copy-on-write branching are planned.", detail: "None of these cache-virtualization features is released in v0.5.3." },
    { title: "分页 KV、块共享与 CoW", summary: "分页分配、共享块与写时复制分支处于规划阶段。", detail: "这些缓存虚拟化能力均未在 v0.5.3 发布。" }),
  fact("adaptive-memory-guard", "planned", "future", ["architecture"], [roadmap],
    { title: "Unified adaptive memory guard", summary: "A feedback controller across cache, model pool, and concurrency is planned.", detail: "Released memory probes and pool caps are separate mechanisms and must not be described as this guard." },
    { title: "统一自适应内存守卫", summary: "跨缓存、模型池与并发的反馈控制器处于规划阶段。", detail: "已发布的内存探针和模型池上限是独立机制，不能称为该守卫。" }),
  fact("sampling-core", "released", "0.1.0", ["models", "choosing-a-model"], [release, tagged("MacMLXCore/Sources/MacMLXCore/Managers/ModelParametersStore.swift")],
    { title: "Temperature and top-p", summary: "Temperature and nucleus top-p sampling are released controls.", detail: "These are the current exposed core sampling controls." },
    { title: "temperature 与 top-p", summary: "temperature 与核采样 top-p 是已发布参数。", detail: "它们是当前已开放的核心采样控制。" }),
  fact("sampling-expanded", "planned", "future", ["models", "choosing-a-model"], [roadmap, main("MacMLXCore/Sources/MacMLXCore/Managers/ModelParametersStore.swift")],
    { title: "Expanded sampling controls", summary: "top-k, min-p, penalties, per-request seed, and exposed KV quantization are planned.", detail: "DeepSeek expert-routing top-k is an internal architecture operation and is unrelated to user sampling top-k." },
    { title: "扩展采样控制", summary: "top-k、min-p、惩罚项、逐请求 seed 与开放 KV 量化处于规划阶段。", detail: "DeepSeek 专家路由 top-k 是架构内部操作，与用户采样 top-k 无关。" }),
]);

function comparisonCell(text, sourceFactIds) {
  return Object.freeze({ text, sourceFactIds: Object.freeze(sourceFactIds) });
}

export const macmlxComparisonProfile = Object.freeze({
  en: Object.freeze({
    platform: comparisonCell("Apple Silicon macOS 14 or later", ["platform-installation"]),
    runtime: comparisonCell("Swift in-process inference through Apple MLX; the default path requires no Python runtime", ["swift-in-process", "no-python-default"]),
    models: comparisonCell("Supported MLX language, vision, embedding, and LoRA workflows", ["vlm-14-families", "embeddings", "lora"]),
    interfaces: comparisonCell("SwiftUI app, macmlx CLI, compatible HTTP APIs, and MCP surfaces", ["shared-core", "openai-compat", "mcp-server"]),
    focus: comparisonCell("Shared MacMLXCore implementation across app, CLI, and local server surfaces", ["shared-core"]),
  }),
  "zh-Hans": Object.freeze({
    platform: comparisonCell("Apple 芯片 macOS 14 或更高版本", ["platform-installation"]),
    runtime: comparisonCell("通过 Apple MLX 运行 Swift 进程内推理；默认路径不需要 Python 运行时", ["swift-in-process", "no-python-default"]),
    models: comparisonCell("受支持的 MLX 语言、视觉、嵌入与 LoRA 工作流", ["vlm-14-families", "embeddings", "lora"]),
    interfaces: comparisonCell("SwiftUI 应用、macmlx CLI、兼容 HTTP API 与 MCP 接口", ["shared-core", "openai-compat", "mcp-server"]),
    focus: comparisonCell("应用、CLI 与本地服务入口共享 MacMLXCore 实现", ["shared-core"]),
  }),
});
