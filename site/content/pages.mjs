import { releases } from "./releases.mjs";

const b = (en, zh) => Object.freeze({ en, "zh-Hans": zh });
const illustration = (src, enAlt, zhAlt, enCaption, zhCaption) => Object.freeze({ type: "illustration", src, width: 2048, height: 1152, alt: b(enAlt, zhAlt), caption: b(enCaption, zhCaption) });
const facts = (en, zh, factIds) => Object.freeze({ type: "facts", heading: b(en, zh), factIds: Object.freeze(factIds) });
const sources = (factIds = [], competitorIds = [], releaseIds = []) => Object.freeze({ type: "sources", heading: b("Official sources", "官方来源"), factIds: Object.freeze(factIds), competitorIds: Object.freeze(competitorIds), releaseIds: Object.freeze(releaseIds) });

function page(id, path, title, description, directAnswer, eyebrow, relatedIds, blocks) {
  const localized = {
    en: Object.freeze({ title: title[0], description: description[0], directAnswer: directAnswer[0], eyebrow: eyebrow[0] }),
    "zh-Hans": Object.freeze({ title: title[1], description: description[1], directAnswer: directAnswer[1], eyebrow: eyebrow[1] }),
  };
  return Object.freeze({
    id,
    kind: "article",
    paths: Object.freeze({ en: path, "zh-Hans": `/zh${path}` }),
    lastVerified: "2026-07-10",
    relatedIds: Object.freeze(relatedIds),
    blocks: Object.freeze([...blocks, Object.freeze({ type: "related", heading: b("Related pages", "相关页面"), relatedIds: Object.freeze(relatedIds) })]),
    ...localized,
  });
}

const apiTable = Object.freeze({
  type: "table",
  heading: b("Endpoint compatibility matrix", "端点兼容性矩阵"),
  caption: b("v0.5.3 endpoint families and their explicit boundaries", "v0.5.3 端点家族及明确边界"),
  headers: Object.freeze({
    en: Object.freeze(["Surface", "Endpoints", "Compatibility boundary"]),
    "zh-Hans": Object.freeze(["接口", "端点", "兼容边界"]),
  }),
  rows: Object.freeze({
    en: Object.freeze([
      Object.freeze(["OpenAI chat", "POST /v1/chat/completions", "Compatible; streaming supported"]),
      Object.freeze(["OpenAI legacy completions", "POST /v1/completions", "Compatible legacy completion shape"]),
      Object.freeze(["OpenAI model listing", "GET /v1/models", "Compatible listing; load/unload are macMLX extensions under /x/models"]),
      Object.freeze(["OpenAI embeddings", "POST /v1/embeddings", "Compatible embeddings shape; model suitability still matters"]),
      Object.freeze(["Anthropic", "POST /v1/messages", "Messages API only, including streaming; not the full Anthropic API"]),
      Object.freeze(["Ollama", "/api/version, /api/tags, /api/show, /api/chat, /api/generate", "Selected endpoints since v0.3.7; not a drop-in replacement"]),
      Object.freeze(["Rerank", "POST /v1/rerank", "macMLX bi-encoder cosine MVP; not a cross-encoder"]),
      Object.freeze(["MCP", "macmlx mcp serve and MCP client pool", "Server released v0.5.0; client pool v0.5.3; chat tool routing is development"]),
    ]),
    "zh-Hans": Object.freeze([
      Object.freeze(["OpenAI 聊天", "POST /v1/chat/completions", "兼容并支持流式响应"]),
      Object.freeze(["OpenAI 传统补全", "POST /v1/completions", "兼容传统补全结构"]),
      Object.freeze(["OpenAI 模型列表", "GET /v1/models", "兼容列表；加载/卸载是 /x/models 下的 macMLX 扩展"]),
      Object.freeze(["OpenAI 嵌入", "POST /v1/embeddings", "兼容嵌入结构；仍需选择合适模型"]),
      Object.freeze(["Anthropic", "POST /v1/messages", "仅 Messages API（含流式），不是完整 Anthropic API"]),
      Object.freeze(["Ollama", "/api/version、/api/tags、/api/show、/api/chat、/api/generate", "自 v0.3.7 起支持选定端点，不是完整替代"]),
      Object.freeze(["重排", "POST /v1/rerank", "macMLX 双编码器余弦 MVP，不是交叉编码器"]),
      Object.freeze(["MCP", "macmlx mcp serve 与 MCP 客户端池", "服务端于 v0.5.0 发布；客户端池于 v0.5.3 发布；聊天工具路由仍在开发"]),
    ]),
  }),
});

const choosingTable = Object.freeze({
  type: "table",
  heading: b("A practical sizing sequence", "实用规模选择顺序"),
  caption: b("Evidence-backed categories to check before downloading", "下载前应核对的有证据类别"),
  headers: Object.freeze({ en: Object.freeze(["Factor", "Start with", "Why it matters"]), "zh-Hans": Object.freeze(["因素", "起步建议", "重要原因"]) }),
  rows: Object.freeze({
    en: Object.freeze([
      Object.freeze(["Task", "Text, code, embedding, or vision", "The task determines the required architecture family"]),
      Object.freeze(["Parameter count", "The smallest model that meets the task", "Weights dominate baseline memory"]),
      Object.freeze(["Quantization", "A supported 4-bit or 8-bit checkpoint", "Lower precision usually reduces weight memory"]),
      Object.freeze(["Context", "The shortest useful context", "Longer context grows KV-cache pressure"]),
      Object.freeze(["Concurrency", "One loaded model first", "Pools and simultaneous work add memory pressure"]),
      Object.freeze(["Headroom", "Leave room for macOS and the workload", "Near-capacity operation is less resilient"]),
    ]),
    "zh-Hans": Object.freeze([
      Object.freeze(["任务", "文本、代码、嵌入或视觉", "任务决定所需架构家族"]),
      Object.freeze(["参数量", "满足任务的最小模型", "权重决定基础内存占用"]),
      Object.freeze(["量化", "受支持的 4-bit 或 8-bit 检查点", "较低精度通常减少权重内存"]),
      Object.freeze(["上下文", "满足需求的最短上下文", "更长上下文会增加 KV 缓存压力"]),
      Object.freeze(["并发", "先从一个已加载模型开始", "模型池与同时工作会增加内存压力"]),
      Object.freeze(["余量", "为 macOS 与工作负载留出空间", "接近容量上限时韧性更差"]),
    ]),
  }),
});

const vlmTable = Object.freeze({
  type: "table",
  heading: b("Vision checkpoint checklist", "视觉检查点核对表"),
  caption: b("The model library detects 14 VLM model_type families; each checkpoint still needs these checks", "模型库检测 14 个 VLM model_type 家族；每个检查点仍需完成以下核对"),
  headers: Object.freeze({ en: Object.freeze(["Category", "Question", "Boundary"]), "zh-Hans": Object.freeze(["类别", "问题", "边界"]) }),
  rows: Object.freeze({
    en: Object.freeze([
      Object.freeze(["Architecture", "Is the exact model_type among the detected families?", "A generic MLX label is not enough"]),
      Object.freeze(["Processor", "Does the checkpoint include compatible image preprocessing?", "Processor variants can differ within a family"]),
      Object.freeze(["Quantization", "Is this quantized checkpoint supported by its architecture?", "Support is not universal across weights"]),
      Object.freeze(["Memory", "Do weights, image tokens, activations, and cache fit?", "Vision adds workload beyond text weights"]),
      Object.freeze(["Serving path", "Does the selected API path accept vision input?", "Development batching remains text-only"]),
    ]),
    "zh-Hans": Object.freeze([
      Object.freeze(["架构", "确切 model_type 是否在检测家族中？", "仅标注 MLX 并不足够"]),
      Object.freeze(["处理器", "检查点是否包含兼容图像预处理？", "同一家族内处理器变体也可能不同"]),
      Object.freeze(["量化", "该量化检查点是否受其架构支持？", "权重兼容并非通用"]),
      Object.freeze(["内存", "权重、图像词元、激活值与缓存是否可容纳？", "视觉负载超出纯文本权重"]),
      Object.freeze(["服务路径", "所选 API 路径是否接受视觉输入？", "开发中批处理仍仅限文本"]),
    ]),
  }),
});

const apiFactIds = ["openai-compat", "anthropic-messages", "ollama-selected", "mcp-server", "mcp-client-pool", "integrated-tool-routing", "embeddings", "rerank-bi-encoder", "continuous-batching"];
const architectureFactIds = ["swift-in-process", "unified-memory", "shared-core", "no-python-default", "tiered-cache", "trie-lcp", "model-pool", "continuous-batching", "fixed-prefill-throttle", "paged-kv", "adaptive-memory-guard"];
const modelFactIds = ["unified-memory", "model-pool", "lora", "vlm-14-families", "deepseek-v32", "sampling-core", "sampling-expanded"];
const comparisonProfileFactIds = ["platform-installation", "swift-in-process", "no-python-default", "vlm-14-families", "embeddings", "lora", "shared-core", "openai-compat", "mcp-server"];
const releaseSourceFactIds = [...new Set(releases.flatMap((item) => [
  ...item.shippedFactIds,
  ...item.limitationFactIds,
  ...item.developmentFactIds,
  ...item.plannedFactIds,
]))];

export const pages = Object.freeze([
  page("architecture", "/architecture/", ["How macMLX runs models on Apple Silicon", "macMLX 如何在 Apple 芯片上运行模型"], ["An evidence-backed guide to the Swift in-process engine, shared core, unified memory, cache tiers, and roadmap boundaries.", "以证据说明 Swift 进程内引擎、共享核心、统一内存、缓存分层与路线图边界。"], ["The released default engine loads, generates, caches, and serves MLX models inside a Swift process. The app and CLI share MacMLXCore code, while separate processes keep separate in-memory engine instances. Post-tag batching and prefix work is clearly separated from v0.5.3.", "已发布的默认引擎在 Swift 进程内加载、生成、缓存并服务 MLX 模型。应用与 CLI 共享 MacMLXCore 代码，但不同进程各自保有内存中引擎实例；标签后的批处理与前缀工作与 v0.5.3 明确区分。"], ["Architecture", "架构"], ["api-compatibility", "models", "releases", "compare-omlx"], [
    illustration("/assets/images/generated/macmlx-shared-core.webp", "Diagram of MacMLXCore shared by the app, CLI, and local API while each process owns its engine instance", "MacMLXCore 由应用、CLI 与本地 API 共享代码，而各进程拥有自己的引擎实例示意图", "One code core across product surfaces; in-memory state remains process-local.", "多个产品入口共享一套核心代码；内存状态仍属于各自进程。"),
    facts("Released architecture and current boundaries", "已发布架构与当前边界", architectureFactIds),
    illustration("/assets/images/generated/macmlx-unified-memory.webp", "Apple Silicon unified memory serving CPU orchestration and integrated GPU MLX compute", "Apple 芯片统一内存服务 CPU 编排与集成 GPU MLX 计算示意图", "Unified memory avoids an explicit discrete-GPU copy boundary, but capacity remains finite.", "统一内存避免独立 GPU 的显式复制边界，但容量仍然有限。"),
    sources(architectureFactIds),
  ]),
  page("api-compatibility", "/api-compatibility/", ["Local API compatibility, endpoint by endpoint", "逐端点说明本地 API 兼容性"], ["A precise matrix for OpenAI, Anthropic Messages, selected Ollama endpoints, MCP, embeddings, rerank, and batching limits.", "精确列出 OpenAI、Anthropic Messages、选定 Ollama 端点、MCP、嵌入、重排与批处理限制。"], ["macMLX exposes several local compatibility surfaces, but compatibility is deliberately scoped by endpoint family. Model load and unload are macMLX extensions, Anthropic support is Messages-only, and Ollama support covers five selected endpoints. Clients should consult this matrix before assuming provider-wide compatibility.", "macMLX 提供多种本地兼容接口，但兼容范围按端点家族明确界定。模型加载与卸载属于 macMLX 扩展，Anthropic 仅支持 Messages，Ollama 覆盖五个选定端点。客户端不应在查看矩阵前假设提供方级别的完整兼容。"], ["API compatibility", "API 兼容性"], ["architecture", "faq", "release-v0-5-3", "compare-ollama"], [
    facts("Compatibility facts", "兼容性事实", apiFactIds),
    apiTable,
    Object.freeze({ type: "code", heading: b("OpenAI-compatible chat example", "OpenAI 兼容聊天示例"), intro: b("Call the local server with an explicit model and message.", "使用明确的模型与消息调用本地服务。"), language: "sh", code: "curl http://localhost:8000/v1/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"model\":\"your-mlx-model\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'" }),
    sources(apiFactIds),
  ]),
  page("models", "/models/", ["Choose models by task and memory", "按任务与内存选择模型"], ["A model hub for MLX checkpoint selection, quantization, vision compatibility, adapters, sampling, and large-model boundaries.", "涵盖 MLX 检查点选择、量化、视觉兼容、适配器、采样与大模型边界的模型中心。"], ["There is no universally best model for every Mac. Choose the smallest supported MLX checkpoint that meets the task, then leave unified-memory headroom for macOS, context, cache, and any concurrent models. Model quality still depends on the task and checkpoint.", "不存在适合每台 Mac 的通用最佳模型。请选择满足任务的最小受支持 MLX 检查点，并为 macOS、上下文、缓存与并发模型保留统一内存余量。模型质量仍取决于任务与检查点。"], ["Model guides", "模型指南"], ["choosing-a-model", "vision-language-models", "architecture", "faq", "compare-lm-studio"], [
    illustration("/assets/images/generated/macmlx-inference-pipeline.webp", "Inference pipeline from a supported MLX checkpoint through preprocessing, generation, cache, and local interfaces", "从受支持 MLX 检查点经过预处理、生成、缓存到本地接口的推理管线示意图", "Compatibility depends on the checkpoint architecture, processor, task, and serving path.", "兼容性取决于检查点架构、处理器、任务与服务路径。"),
    facts("Model capability boundaries", "模型能力边界", modelFactIds),
    sources(modelFactIds),
  ]),
  page("choosing-a-model", "/models/choosing-a-model/", ["How to choose an MLX model", "如何选择 MLX 模型"], ["A practical table for task, parameter count, quantization, context, concurrency, and unified-memory headroom.", "按任务、参数量、量化、上下文、并发与统一内存余量选择模型的实用表格。"], ["Start with the task and exact architecture support, then size weights, context cache, runtime overhead, and macOS headroom. If the estimate approaches physical memory, choose a smaller or more strongly quantized checkpoint. This conservative sequence avoids treating nominal model size as total runtime memory.", "先确认任务与确切架构支持，再估算权重、上下文缓存、运行时开销与 macOS 余量。若估算接近物理内存，应选择更小或量化程度更高的检查点。这一保守顺序避免把名义模型大小当作运行时总内存。"], ["Model selection", "模型选择"], ["models", "vision-language-models", "architecture"], [facts("Sizing facts", "规模选择事实", ["unified-memory", "model-pool", "tiered-cache", "sampling-core", "sampling-expanded"]), choosingTable, sources(["unified-memory", "model-pool", "tiered-cache", "sampling-core", "sampling-expanded"])]),
  page("vision-language-models", "/models/vision-language-models/", ["Vision-language model support", "视觉语言模型支持"], ["What the 14 detected VLM families mean, and what to verify for checkpoint, processor, memory, and serving compatibility.", "解释 14 个检测到的 VLM 家族，并说明检查点、处理器、内存与服务兼容性核对项。"], ["The v0.5.3 model library detects 14 VLM model_type families. That is a family-level signal, not universal checkpoint support: image processors, weight variants, quantization, memory use, and the selected serving path still matter. Use the checklist below before downloading or exposing a checkpoint through an API.", "v0.5.3 模型库检测 14 个 VLM model_type 家族。这是家族级信号，不代表通用检查点支持；图像处理器、权重变体、量化、内存占用与服务路径仍然重要。下载或通过 API 提供检查点前，请使用下方核对表。"], ["Vision models", "视觉模型"], ["models", "choosing-a-model", "release-v0-5-3"], [facts("Vision facts", "视觉事实", ["vlm-14-families", "unified-memory", "continuous-batching"]), vlmTable, sources(["vlm-14-families", "unified-memory", "continuous-batching"])]),
  page("faq", "/faq/", ["macMLX questions, answered", "macMLX 常见问题解答"], ["Eight visible answers covering platform, installation, Python, models, APIs, privacy, vision, large MoE models, and roadmap status.", "八个可见答案，涵盖平台、安装、Python、模型、API、隐私、视觉、大型 MoE 与路线图状态。"], ["These answers use the v0.5.3 release as the baseline and visibly distinguish released, post-tag development, and planned work. Follow the linked technical pages when compatibility depends on a model or endpoint. Each visible answer points back to dated, official evidence.", "以下答案以 v0.5.3 为基线，并明确区分已发布、标签后开发与规划工作。若兼容性取决于模型或端点，请继续查看相关技术页面。每个可见答案都可追溯到带日期的官方证据。"], ["FAQ", "常见问题"], ["api-compatibility", "models", "releases"], [Object.freeze({ type: "faq", heading: b("Eight common questions", "八个常见问题"), faqIds: Object.freeze(["platform-installation", "python", "model-selection", "apis", "privacy", "vlm", "large-moe", "roadmap"]) }), sources(["platform-installation", "no-python-default", "unified-memory", "model-pool", "lora", "vlm-14-families", "openai-compat", "anthropic-messages", "ollama-selected", "swift-in-process", "deepseek-v32", "continuous-batching", "trie-lcp", "paged-kv", "adaptive-memory-guard", "sampling-expanded"])]),
  page("compare", "/compare/", ["Compare local model tools by factual route", "按事实路线对比本地模型工具"], ["A neutral, dated overview of macMLX, Ollama, LM Studio, oMLX, Swama, and SwiftLM with official sources.", "以官方来源中立对比 macMLX、Ollama、LM Studio、oMLX、Swama 与 SwiftLM，并标明日期。"], ["These tools differ in platform reach, core runtime, model workflow, interfaces, and technical focus. The table uses official snapshots and declares no score, winner, or universal best choice. Use the dated dimensions to identify the workflow that matches your requirements.", "这些工具在平台范围、核心运行时、模型工作流、接口与技术重点上有所不同。表格采用官方快照，不给出评分、赢家或通用最佳选择。请根据带日期的维度寻找符合自身需求的工作流。"], ["Comparisons", "产品对比"], ["compare-ollama", "compare-lm-studio", "compare-omlx"], [Object.freeze({ type: "comparison", heading: b("Six factual snapshots", "六个事实快照"), competitorIds: Object.freeze(["ollama", "lm-studio", "omlx", "swama", "swiftlm"]) }), sources(comparisonProfileFactIds, ["ollama", "lm-studio", "omlx", "swama", "swiftlm"])]),
  page("compare-ollama", "/compare/ollama/", ["macMLX and Ollama", "macMLX 与 Ollama"], ["A neutral comparison of macMLX v0.5.3 and Ollama v0.31.2 by platform, runtime, models, interfaces, and focus.", "按平台、运行时、模型、接口与重点中立对比 macMLX v0.5.3 与 Ollama v0.31.2。"], ["macMLX centers on a Swift in-process MLX core for Apple Silicon. Ollama provides a cross-platform service and model workflow, including official MLX support on Apple Silicon. The right fit depends on platform and interface requirements. The comparison below uses July 2026 official snapshots rather than subjective scores.", "macMLX 以 Apple 芯片上的 Swift 进程内 MLX 核心为中心；Ollama 提供跨平台服务与模型工作流，并在 Apple 芯片上官方支持 MLX。选择取决于平台与接口需求。下方对比采用 2026 年 7 月官方快照，而非主观评分。"], ["Comparison", "产品对比"], ["compare", "api-compatibility", "architecture"], [Object.freeze({ type: "comparison", heading: b("Runtime and workflow snapshot", "运行时与工作流快照"), competitorIds: Object.freeze(["ollama"]) }), sources([...comparisonProfileFactIds, "ollama-selected"], ["ollama"])]),
  page("compare-lm-studio", "/compare/lm-studio/", ["macMLX and LM Studio", "macMLX 与 LM Studio"], ["A neutral comparison of macMLX v0.5.3 and LM Studio 0.4.19 Build 2 using official sources.", "依据官方来源中立对比 macMLX v0.5.3 与 LM Studio 0.4.19 Build 2。"], ["macMLX focuses on one Swift in-process MLX core for Apple Silicon. LM Studio spans desktop, headless, CLI, SDK, and local-server workflows across platforms and runtimes; its official MLX engine is Python-based and bundles Python 3.11. The table compares those factual routes without rating either product.", "macMLX 聚焦 Apple 芯片上的一套 Swift 进程内 MLX 核心。LM Studio 跨平台与多运行时覆盖桌面、无头、CLI、SDK 与本地服务工作流；其官方 MLX 引擎基于 Python 并捆绑 Python 3.11。表格只比较这些事实路线，不对任一产品评分。"], ["Comparison", "产品对比"], ["compare", "models", "architecture"], [Object.freeze({ type: "comparison", heading: b("Runtime and product-surface snapshot", "运行时与产品入口快照"), competitorIds: Object.freeze(["lm-studio"]) }), sources(comparisonProfileFactIds, ["lm-studio"])]),
  page("compare-omlx", "/compare/omlx/", ["macMLX and oMLX", "macMLX 与 oMLX"], ["A neutral comparison of macMLX v0.5.3 and stable oMLX v0.4.4, not a release candidate.", "中立对比 macMLX v0.5.3 与稳定版 oMLX v0.4.4，而非候选版本。"], ["Both target MLX inference on Apple Silicon and provide native macOS surfaces. macMLX runs its default engine in-process in Swift; oMLX v0.4.4 uses a Python/FastAPI core with a native SwiftUI menu-bar app and documented continuous batching and paged cache features.", "两者都面向 Apple 芯片上的 MLX 推理并提供原生 macOS 入口。macMLX 默认引擎在 Swift 进程内运行；oMLX v0.4.4 使用 Python/FastAPI 核心与原生 SwiftUI 菜单栏应用，并记录了连续批处理与分页缓存能力。"], ["Comparison", "产品对比"], ["compare", "architecture", "releases"], [Object.freeze({ type: "comparison", heading: b("Engine and serving snapshot", "引擎与服务快照"), competitorIds: Object.freeze(["omlx"]) }), sources([...comparisonProfileFactIds, "continuous-batching", "paged-kv"], ["omlx"])]),
  page("releases", "/releases/", ["Release status without roadmap blur", "不混淆路线图的版本状态"], ["A v0.5.3 release summary that separates shipped capabilities, current limitations, post-tag development, and planned work.", "v0.5.3 版本摘要，区分已交付能力、当前限制、标签后开发与规划工作。"], ["The current release baseline is v0.5.3 from 2026-07-08. GitHub Releases and the immutable tagged changelog are authoritative; this hub makes the boundary with post-tag main and future plans easier to read. Capability labels below remain visible so roadmap work is not mistaken for shipped behavior.", "当前发布基线是 2026-07-08 的 v0.5.3。GitHub Releases 与不可变标签更新日志是权威来源；本中心让标签后 main 分支与未来计划的边界更易读。下方能力标签保持可见，避免把路线图工作误认为已交付行为。"], ["Releases", "版本"], ["release-v0-5-3", "architecture", "api-compatibility"], [Object.freeze({ type: "release", heading: b("Current release", "当前版本"), releaseIds: Object.freeze(["v0-5-3"]) }), sources(releaseSourceFactIds, [], ["v0-5-3"])]),
  page("release-v0-5-3", "/releases/v0-5-3/", ["macMLX v0.5.3", "macMLX v0.5.3"], ["What shipped on 2026-07-08, current limitations, post-tag development, and planned engine work.", "说明 2026-07-08 已交付内容、当前限制、标签后开发与规划中的引擎工作。"], ["v0.5.3 is the audited release baseline. It ships the API, embedding, cache, model-pool, MCP, LoRA, VLM, and DeepSeek work listed below; continuous batching, trie prefix reuse, paged KV, and expanded controls retain their development or planned labels. GitHub's tagged release and changelog remain the authoritative distribution record.", "v0.5.3 是经审计的发布基线。它交付下列 API、嵌入、缓存、模型池、MCP、LoRA、VLM 与 DeepSeek 工作；连续批处理、Trie 前缀复用、分页 KV 与扩展控制仍保留开发中或规划标签。GitHub 标签版本与更新日志仍是权威发布记录。"], ["Release", "版本"], ["releases", "api-compatibility", "vision-language-models"], [Object.freeze({ type: "release", heading: b("Shipped, limited, developing, and planned", "已交付、受限、开发中与规划中"), releaseIds: Object.freeze(["v0-5-3"]) }), sources(releaseSourceFactIds, [], ["v0-5-3"])]),
]);
