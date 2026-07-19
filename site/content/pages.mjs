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
    lastVerified: "2026-07-19",
    relatedIds: Object.freeze(relatedIds),
    blocks: Object.freeze([...blocks, Object.freeze({ type: "related", heading: b("Related pages", "相关页面"), relatedIds: Object.freeze(relatedIds) })]),
    ...localized,
  });
}

const apiTable = Object.freeze({
  type: "table",
  heading: b("Endpoint compatibility matrix", "端点兼容性矩阵"),
  caption: b("v0.7.0 endpoint families and their explicit boundaries", "v0.7.0 端点家族及明确边界"),
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
      Object.freeze(["Tool loops", "OpenAI, Anthropic, and GUI MCP routes", "Multi-turn tool routing released in v0.6.0"]),
      Object.freeze(["Structured output", "response_format: json_object or supported JSON Schema", "Unsupported schemas and tool/VLM combinations return 400"]),
      Object.freeze(["API compatibility pack", "logit_bias, logprobs, top_logprobs, XTC, per-request LoRA, tools", "Unsupported parameter combinations return 400"]),
      Object.freeze(["KV-cache quantization", "kv_bits, kv_group_size, quantized_kv_start", "Compatible requests only; this does not quantize model weights"]),
    ]),
    "zh-Hans": Object.freeze([
      Object.freeze(["OpenAI 聊天", "POST /v1/chat/completions", "兼容并支持流式响应"]),
      Object.freeze(["OpenAI 传统补全", "POST /v1/completions", "兼容传统补全结构"]),
      Object.freeze(["OpenAI 模型列表", "GET /v1/models", "兼容列表；加载/卸载是 /x/models 下的 macMLX 扩展"]),
      Object.freeze(["OpenAI 嵌入", "POST /v1/embeddings", "兼容嵌入结构；仍需选择合适模型"]),
      Object.freeze(["Anthropic", "POST /v1/messages", "仅 Messages API（含流式），不是完整 Anthropic API"]),
      Object.freeze(["Ollama", "/api/version、/api/tags、/api/show、/api/chat、/api/generate", "自 v0.3.7 起支持选定端点，不是完整替代"]),
      Object.freeze(["重排", "POST /v1/rerank", "macMLX 双编码器余弦 MVP，不是交叉编码器"]),
      Object.freeze(["工具循环", "OpenAI、Anthropic 与 GUI MCP 路由", "多轮工具路由于 v0.6.0 发布"]),
      Object.freeze(["结构化输出", "response_format：json_object 或受支持的 JSON Schema", "不支持的 Schema 以及工具/VLM 组合返回 400"]),
      Object.freeze(["API 兼容功能包", "logit_bias、logprobs、top_logprobs、XTC、逐请求 LoRA、tools", "不支持的参数组合返回 400"]),
      Object.freeze(["KV 缓存量化", "kv_bits、kv_group_size、quantized_kv_start", "仅限兼容请求；这不会量化模型权重"]),
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
      Object.freeze(["Serving path", "Does the selected API path accept vision input?", "Batching is dense-text-only; VLM requests remain serial"]),
    ]),
    "zh-Hans": Object.freeze([
      Object.freeze(["架构", "确切 model_type 是否在检测家族中？", "仅标注 MLX 并不足够"]),
      Object.freeze(["处理器", "检查点是否包含兼容图像预处理？", "同一家族内处理器变体也可能不同"]),
      Object.freeze(["量化", "该量化检查点是否受其架构支持？", "权重兼容并非通用"]),
      Object.freeze(["内存", "权重、图像词元、激活值与缓存是否可容纳？", "视觉负载超出纯文本权重"]),
      Object.freeze(["服务路径", "所选 API 路径是否接受视觉输入？", "批处理仅限稠密文本；VLM 请求保持串行"]),
    ]),
  }),
});

const apiFactIds = ["openai-compat", "anthropic-messages", "ollama-selected", "mcp-server", "mcp-client-pool", "integrated-tool-routing", "embeddings", "rerank-bi-encoder", "continuous-batching", "structured-output", "api-compat-pack", "kv-cache-quantization"];
const architectureFactIds = ["swift-in-process", "unified-memory", "shared-core", "no-python-default", "tiered-cache", "trie-lcp", "model-pool", "continuous-batching", "fixed-prefill-throttle", "speculative-decoding", "kv-cache-quantization", "silicon-activity-panel", "bottleneck-classifier", "silicon-sampling", "paged-kv", "adaptive-memory-guard"];
const modelFactIds = ["unified-memory", "model-pool", "lora", "vlm-14-families", "deepseek-v32", "speculative-decoding", "hf-cache-discovery", "chat-template-overrides", "track-g-tested-models", "internlm3-theoretical", "ocr-recognition", "benchmark-attribution", "sampling-core", "sampling-expanded"];
const comparisonProfileFactIds = ["platform-installation", "swift-in-process", "no-python-default", "vlm-14-families", "embeddings", "lora", "track-g-tested-models", "internlm3-theoretical", "shared-core", "openai-compat", "mcp-server", "structured-output", "integrated-tool-routing", "continuous-batching", "trie-lcp", "speculative-decoding", "silicon-activity-panel", "bottleneck-classifier"];
const releaseSourceFactIds = [...new Set(releases.flatMap((item) => [
  ...item.shippedFactIds,
  ...item.limitationFactIds,
  ...item.developmentFactIds,
  ...item.plannedFactIds,
]))];

function releaseFactIds(id) {
  const item = releases.find((release) => release.id === id);
  if (item === undefined) throw new Error(`unknown release: ${id}`);
  return [...new Set([
    ...item.shippedFactIds,
    ...item.limitationFactIds,
    ...item.developmentFactIds,
    ...item.plannedFactIds,
  ])];
}

export const pages = Object.freeze([
  page("architecture", "/architecture/", ["How macMLX runs models on Apple Silicon", "macMLX 如何在 Apple 芯片上运行模型"], ["An evidence-backed v0.7.0 guide to the Swift engine, shared core, unified memory, batching, sudoless silicon observability, cache tiers, and roadmap boundaries.", "以 v0.7.0 证据说明 Swift 引擎、共享核心、统一内存、批处理、免 sudo 的硅可观测性、缓存分层与路线图边界。"], ["The current v0.7.0 engine runs in Swift, while separate processes keep separate in-memory engine instances. Eligibility-gated continuous batching measured 2.5–3.2× aggregate throughput with four clients; longest-prefix reuse, speculative decoding, and KV-cache quantization are also released. v0.7.0 adds a sudoless silicon Activity panel that attributes the live inference bottleneck. Paged KV, block sharing, and CoW plus an adaptive memory guard remain planned.", "当前 v0.7.0 引擎在 Swift 中运行，不同进程仍各自保有内存中引擎实例。资格门控的连续批处理在四客户端下测得 2.5–3.2× 聚合吞吐量；最长前缀复用、推测解码与 KV 缓存量化也已发布。v0.7.0 新增免 sudo 的硅活动面板，用于归因实时推理瓶颈。分页 KV、块共享、写时复制与自适应内存守卫仍在规划中。"], ["Architecture", "架构"], ["api-compatibility", "models", "releases", "compare-omlx"], [
    illustration("/assets/images/generated/macmlx-shared-core.webp", "Diagram of MacMLXCore shared by the app, CLI, and local API while each process owns its engine instance", "MacMLXCore 由应用、CLI 与本地 API 共享代码，而各进程拥有自己的引擎实例示意图", "One code core across product surfaces; in-memory state remains process-local.", "多个产品入口共享一套核心代码；内存状态仍属于各自进程。"),
    facts("Released architecture and current boundaries", "已发布架构与当前边界", architectureFactIds),
    illustration("/assets/images/generated/macmlx-unified-memory.webp", "Apple Silicon unified memory serving CPU orchestration and integrated GPU MLX compute", "Apple 芯片统一内存服务 CPU 编排与集成 GPU MLX 计算示意图", "Unified memory avoids an explicit discrete-GPU copy boundary, but capacity remains finite.", "统一内存避免独立 GPU 的显式复制边界，但容量仍然有限。"),
    sources(architectureFactIds),
  ]),
  page("api-compatibility", "/api-compatibility/", ["Local API compatibility, endpoint by endpoint", "逐端点说明本地 API 兼容性"], ["A precise v0.7.0 matrix for OpenAI, Anthropic Messages, selected Ollama endpoints, tool loops, structured output, API controls, and serving boundaries.", "精确列出 v0.7.0 的 OpenAI、Anthropic Messages、选定 Ollama 端点、工具循环、结构化输出、API 控制项与服务边界。"], ["The current v0.7.0 APIs expose released tool loops, structured output, compatibility controls, and KV-cache quantization across scoped local surfaces. Model management under /x/models remains a macMLX extension; Anthropic support is Messages-only; Ollama covers five selected endpoints; rerank remains a bi-encoder MVP. Unsupported schemas and parameter combinations return explicit 400 responses.", "当前 v0.7.0 API 在范围明确的本地接口中提供已发布的工具循环、结构化输出、兼容控制项与 KV 缓存量化。/x/models 下的模型管理仍是 macMLX 扩展；Anthropic 仅支持 Messages；Ollama 覆盖五个选定端点；重排仍是双编码器 MVP。不支持的 Schema 与参数组合会明确返回 400。"], ["API compatibility", "API 兼容性"], ["architecture", "faq", "release-v0-7-0", "compare-ollama"], [
    facts("Compatibility facts", "兼容性事实", apiFactIds),
    apiTable,
    Object.freeze({ type: "code", heading: b("OpenAI-compatible chat example", "OpenAI 兼容聊天示例"), intro: b("Call the local server with an explicit model and message.", "使用明确的模型与消息调用本地服务。"), language: "sh", code: "curl http://localhost:8000/v1/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"model\":\"your-mlx-model\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'" }),
    sources(apiFactIds),
  ]),
  page("models", "/models/", ["Choose models by task and memory", "按任务与内存选择模型"], ["A v0.7.0 model hub for checkpoint selection, cache discovery, templates, tested Track G models, OCR recognition, vision, sampling, and support boundaries.", "v0.7.0 模型中心，涵盖检查点选择、缓存发现、模板、实测 Track G 模型、OCR 识别、视觉、采样与支持边界。"], ["The current v0.7.0 model hub includes checkpoint-tested Track G models such as Seed-OSS-36B, with results that are checkpoint-specific, not family-wide performance guarantees. InternLM3-8B remains theoretical only because public checkpoints provide tokenizer.model while the Swift path requires tokenizer.json. v0.7.0 adds OCR-model recognition, so a verified checkpoint like GLM-OCR earns an OCR badge distinct from Vision. Choose by exact architecture, tokenizer, template, memory, and task.", "当前 v0.7.0 模型中心包含经检查点实测的 Track G 模型（如 Seed-OSS-36B），结果仅适用于特定检查点，并非家族级性能保证。InternLM3-8B 仍仅为理论支持，因为公开检查点提供 tokenizer.model，而 Swift 路径需要 tokenizer.json。v0.7.0 新增 OCR 模型识别，因此像 GLM-OCR 这样经过验证的检查点会获得区别于 Vision 的 OCR 徽章。请依据确切架构、分词器、模板、内存与任务进行选择。"], ["Model guides", "模型指南"], ["choosing-a-model", "vision-language-models", "architecture", "faq", "compare-lm-studio"], [
    illustration("/assets/images/generated/macmlx-inference-pipeline.webp", "Inference pipeline from a supported MLX checkpoint through preprocessing, generation, cache, and local interfaces", "从受支持 MLX 检查点经过预处理、生成、缓存到本地接口的推理管线示意图", "Compatibility depends on the checkpoint architecture, processor, task, and serving path.", "兼容性取决于检查点架构、处理器、任务与服务路径。"),
    facts("Model capability boundaries", "模型能力边界", modelFactIds),
    sources(modelFactIds),
  ]),
  page("choosing-a-model", "/models/choosing-a-model/", ["How to choose an MLX model", "如何选择 MLX 模型"], ["A practical v0.7.0 table for architecture, local discovery, parameter count, quantization, context, concurrency, and unified-memory headroom.", "v0.7.0 实用表格，涵盖架构、本地发现、参数量、量化、上下文、并发与统一内存余量。"], ["Start with the task and exact architecture support, then verify tokenizer and checkpoint compatibility, including models discovered in Hugging Face cache roots. Size weights, context cache, runtime overhead, and macOS headroom; use compatible KV-cache quantization only where appropriate. If the estimate approaches physical memory, choose a smaller or more strongly weight-quantized checkpoint.", "先从任务与确切架构支持开始，再核对分词器和检查点兼容性，包括在 Hugging Face 缓存根目录中发现的模型。估算权重、上下文缓存、运行时开销与 macOS 余量；仅在兼容场景使用 KV 缓存量化。若估算接近物理内存，应选择更小或权重量化程度更高的检查点。"], ["Model selection", "模型选择"], ["models", "vision-language-models", "architecture"], [facts("Sizing facts", "规模选择事实", ["unified-memory", "model-pool", "tiered-cache", "kv-cache-quantization", "hf-cache-discovery", "sampling-core", "sampling-expanded"]), choosingTable, sources(["unified-memory", "model-pool", "tiered-cache", "kv-cache-quantization", "hf-cache-discovery", "sampling-core", "sampling-expanded"])]),
  page("vision-language-models", "/models/vision-language-models/", ["Vision-language model support", "视觉语言模型支持"], ["What the 14 detected VLM families mean in v0.7.0, and what to verify for checkpoint, processor, memory, and serving compatibility.", "解释 v0.7.0 中 14 个检测到的 VLM 家族，并说明检查点、处理器、内存与服务兼容性核对项。"], ["The v0.7.0 model library detects exactly 14 VLM model_type families. This is a family-level signal, not universal checkpoint support: image processors, weight variants, quantization, memory use, and serving paths still matter. Dedicated OCR checkpoints such as GLM-OCR load through this same VLM path and earn an OCR badge. Eligibility-gated batching remains dense-text-only, so VLM requests stay serial.", "v0.7.0 模型库检测到恰好 14 个 VLM model_type 家族。这是家族级信号，不代表通用检查点支持；图像处理器、权重变体、量化、内存占用与服务路径仍然重要。像 GLM-OCR 这样的专用 OCR 检查点通过同一 VLM 路径加载，并获得 OCR 徽章。资格门控批处理仅限稠密文本，因此 VLM 请求保持串行。"], ["Vision models", "视觉模型"], ["models", "choosing-a-model", "release-v0-7-0"], [facts("Vision facts", "视觉事实", ["vlm-14-families", "unified-memory", "continuous-batching"]), vlmTable, sources(["vlm-14-families", "unified-memory", "continuous-batching"])]),
  page("faq", "/faq/", ["macMLX questions, answered", "macMLX 常见问题解答"], ["Eight v0.7.0 answers covering platform, installation, Python, models, APIs, privacy, vision, large MoE models, and roadmap status.", "八个 v0.7.0 答案，涵盖平台、安装、Python、模型、API、隐私、视觉、大型 MoE 与路线图状态。"], ["These answers use v0.7.0 as the audited baseline and distinguish released capabilities from planned work. API controls retain provider, schema, and combination boundaries; VLM and large-model answers retain checkpoint-specific caveats. Follow the linked technical pages for details. Every visible claim points to dated official evidence.", "以下答案以 v0.7.0 为审计基线，并区分已发布能力与规划工作。API 控制项保留提供方、Schema 与组合边界；VLM 与大型模型答案保留检查点特定限制。详情请查看相关技术页面。每个可见声明都指向带日期的官方证据。"], ["FAQ", "常见问题"], ["api-compatibility", "models", "releases", "release-v0-5-3"], [Object.freeze({ type: "faq", heading: b("Eight common questions", "八个常见问题"), faqIds: Object.freeze(["platform-installation", "python", "model-selection", "apis", "privacy", "vlm", "large-moe", "roadmap"]) }), sources(["platform-installation", "no-python-default", "unified-memory", "model-pool", "lora", "vlm-14-families", "openai-compat", "anthropic-messages", "ollama-selected", "integrated-tool-routing", "structured-output", "api-compat-pack", "kv-cache-quantization", "swift-in-process", "deepseek-v32", "track-g-tested-models", "continuous-batching", "trie-lcp", "paged-kv", "adaptive-memory-guard", "sampling-expanded"])]),
  page("compare", "/compare/", ["Compare local model tools by factual route", "按事实路线对比本地模型工具"], ["A neutral, dated v0.7.0 overview of macMLX, Ollama, LM Studio, oMLX, Swama, and SwiftLM with official sources.", "以官方来源中立对比 v0.7.0 macMLX、Ollama、LM Studio、oMLX、Swama 与 SwiftLM，并标明日期。"], ["The macMLX v0.7.0 snapshot and the unchanged competitor snapshots differ in platform reach, runtime, model workflow, interfaces, and focus. The table uses official evidence and declares no score, winner, or universal best choice. Use the dated dimensions to identify the workflow that matches your requirements.", "macMLX v0.7.0 快照与未变更的竞品快照在平台范围、运行时、模型工作流、接口与重点上存在差异。表格采用官方证据，不给出评分、赢家或通用最佳选择。请根据带日期的维度寻找符合自身需求的工作流。"], ["Comparisons", "产品对比"], ["compare-ollama", "compare-lm-studio", "compare-omlx"], [Object.freeze({ type: "comparison", heading: b("Six factual snapshots", "六个事实快照"), competitorIds: Object.freeze(["ollama", "lm-studio", "omlx", "swama", "swiftlm"]) }), sources(comparisonProfileFactIds, ["ollama", "lm-studio", "omlx", "swama", "swiftlm"])]),
  page("compare-ollama", "/compare/ollama/", ["macMLX and Ollama", "macMLX 与 Ollama"], ["A neutral comparison of macMLX v0.7.0 and Ollama v0.31.2 by platform, runtime, models, interfaces, and focus.", "按平台、运行时、模型、接口与重点中立对比 macMLX v0.7.0 与 Ollama v0.31.2。"], ["macMLX v0.7.0 centers on a Swift in-process MLX core for Apple Silicon. Ollama provides a cross-platform service and model workflow, including official MLX support on Apple Silicon. The right fit depends on platform and interface requirements. The comparison below uses July 2026 official snapshots rather than subjective scores.", "macMLX v0.7.0 以 Apple 芯片上的 Swift 进程内 MLX 核心为中心；Ollama 提供跨平台服务与模型工作流，并在 Apple 芯片上官方支持 MLX。选择取决于平台与接口需求。下方对比采用 2026 年 7 月官方快照，而非主观评分。"], ["Comparison", "产品对比"], ["compare", "api-compatibility", "architecture"], [Object.freeze({ type: "comparison", heading: b("Runtime and workflow snapshot", "运行时与工作流快照"), competitorIds: Object.freeze(["ollama"]) }), sources([...comparisonProfileFactIds, "ollama-selected"], ["ollama"])]),
  page("compare-lm-studio", "/compare/lm-studio/", ["macMLX and LM Studio", "macMLX 与 LM Studio"], ["A neutral comparison of macMLX v0.7.0 and LM Studio 0.4.19 Build 2 using official sources.", "依据官方来源中立对比 macMLX v0.7.0 与 LM Studio 0.4.19 Build 2。"], ["macMLX v0.7.0 focuses on one Swift in-process MLX core for Apple Silicon. LM Studio spans desktop, headless, CLI, SDK, and local-server workflows across platforms and runtimes; its official MLX engine is Python-based and bundles Python 3.11. The table compares those factual routes without rating either product.", "macMLX v0.7.0 聚焦 Apple 芯片上的一套 Swift 进程内 MLX 核心。LM Studio 跨平台与多运行时覆盖桌面、无头、CLI、SDK 与本地服务工作流；其官方 MLX 引擎基于 Python 并捆绑 Python 3.11。表格只比较这些事实路线，不对任一产品评分。"], ["Comparison", "产品对比"], ["compare", "models", "architecture"], [Object.freeze({ type: "comparison", heading: b("Runtime and product-surface snapshot", "运行时与产品入口快照"), competitorIds: Object.freeze(["lm-studio"]) }), sources(comparisonProfileFactIds, ["lm-studio"])]),
  page("compare-omlx", "/compare/omlx/", ["macMLX and oMLX", "macMLX 与 oMLX"], ["A neutral comparison of macMLX v0.7.0 and stable oMLX v0.4.4, not a release candidate.", "中立对比 macMLX v0.7.0 与稳定版 oMLX v0.4.4，而非候选版本。"], ["Both target MLX inference on Apple Silicon and provide native macOS surfaces. macMLX v0.7.0 runs its default engine in-process in Swift; oMLX v0.4.4 uses a Python/FastAPI core with a native SwiftUI menu-bar app and documented continuous batching and paged cache features.", "两者都面向 Apple 芯片上的 MLX 推理并提供原生 macOS 入口。macMLX v0.7.0 默认引擎在 Swift 进程内运行；oMLX v0.4.4 使用 Python/FastAPI 核心与原生 SwiftUI 菜单栏应用，并记录了连续批处理与分页缓存能力。"], ["Comparison", "产品对比"], ["compare", "architecture", "releases"], [Object.freeze({ type: "comparison", heading: b("Engine and serving snapshot", "引擎与服务快照"), competitorIds: Object.freeze(["omlx"]) }), sources([...comparisonProfileFactIds, "continuous-batching", "paged-kv"], ["omlx"])]),
  page("releases", "/releases/", ["Release status without roadmap blur", "不混淆路线图的版本状态"], ["A current v0.7.0 hub with historical v0.6.2 and v0.5.3 records that separates shipped capabilities, limitations, and planned work.", "当前 v0.7.0 版本中心，附 v0.6.2 与 v0.5.3 历史记录，区分已交付能力、限制与规划工作。"], ["The current audited release is v0.7.0 from 2026-07-18; v0.6.2 and v0.5.3 remain available as historical records. GitHub Releases, immutable tagged changelogs, and the tagged model-support guide are authoritative. This hub renders every record and preserves visible capability labels so later features and future roadmap work are never attributed to a historical release.", "当前审计版本是 2026-07-18 发布的 v0.7.0；v0.6.2 与 v0.5.3 保留为历史记录。GitHub Releases、不可变标签更新日志与标签版模型支持指南是权威来源。本中心呈现每条记录并保留可见能力标签，避免把后续功能或未来路线图归入某个历史版本。"], ["Releases", "版本"], ["release-v0-7-0", "release-v0-6-2", "release-v0-5-3", "architecture", "api-compatibility"], [Object.freeze({ type: "release", heading: b("Current and historical releases", "当前与历史版本"), releaseIds: Object.freeze(["v0-7-0", "v0-6-2", "v0-5-3"]) }), sources(releaseSourceFactIds, [], ["v0-7-0", "v0-6-2", "v0-5-3"])]),
  page("release-v0-7-0", "/releases/v0-7-0/", ["macMLX v0.7.0", "macMLX v0.7.0"], ["The current release: a sudoless silicon Activity panel, live inference-bottleneck attribution, benchmark bottleneck attribution, and OCR-model recognition.", "当前版本：免 sudo 的硅活动面板、实时推理瓶颈归因、基准瓶颈归因与 OCR 模型识别。"], ["v0.7.0 is the current audited baseline. It adds a sudoless silicon-metrics Activity panel, an in-process inference-bottleneck classifier, benchmark bottleneck attribution, and OCR-model recognition, while retaining explicit estimated-versus-measured, availability, and checkpoint boundaries. The tagged release and changelog are authoritative for shipped observability and OCR work; the SSD KV-cache cold-tier waves on main are development, not part of this tagged release.", "v0.7.0 是当前审计基线。它新增免 sudo 的硅指标活动面板、进程内推理瓶颈分类器、基准瓶颈归因与 OCR 模型识别，同时保留明确的估算与实测、可用性与检查点边界。标签版发布记录与更新日志是已发布可观测能力与 OCR 工作的权威依据；main 分支上的 SSD KV 缓存冷层各阶段属于开发中，不属于此标签版本。"], ["Release", "版本"], ["releases", "models", "release-v0-6-2"], [Object.freeze({ type: "release", heading: b("Current v0.7.0 release", "当前 v0.7.0 版本"), releaseIds: Object.freeze(["v0-7-0"]) }), sources(releaseFactIds("v0-7-0"), [], ["v0-7-0"])]),
  page("release-v0-6-2", "/releases/v0-6-2/", ["macMLX v0.6.2 (historical)", "macMLX v0.6.2（历史版本）"], ["A historical record of what shipped on 2026-07-11: tool loops, batching, longest-prefix reuse, structured output, speculative decoding, API controls, and Track G model work.", "2026-07-11 已交付内容的历史记录：工具循环、批处理、最长前缀复用、结构化输出、推测解码、API 控制项与 Track G 模型工作。"], ["v0.6.2 is an explicitly historical audited record. It covers the v0.6 agent backend, Track G models, and per-model chat-template overrides without importing later v0.7.0 silicon-metrics work. Its tagged GitHub release, changelog, and model-support guide remain authoritative; this page intentionally keeps only the v0.6.2 release block, its limitations, and the planned work recorded at that tag.", "v0.6.2 是明确标注的历史审计记录。它涵盖 v0.6 智能体后端、Track G 模型与逐模型聊天模板覆盖，不引入后续 v0.7.0 硅指标工作。其 GitHub 标签发布、更新日志与模型支持指南仍是权威依据；本页有意只保留 v0.6.2 版本块、其限制与该标签时记录的规划工作。"], ["Historical release", "历史版本"], ["releases", "api-compatibility", "models"], [Object.freeze({ type: "release", heading: b("Historical v0.6.2 scope", "历史 v0.6.2 范围"), releaseIds: Object.freeze(["v0-6-2"]) }), sources([], [], ["v0-6-2"])]),
  page("release-v0-5-3", "/releases/v0-5-3/", ["macMLX v0.5.3 (historical)", "macMLX v0.5.3（历史版本）"], ["A retrospective summary of what shipped on 2026-07-08, using current audited definitions while preserving the tagged v0.5.3 record as authority.", "对 2026-07-08 已交付内容的回顾性摘要，使用当前审计定义，并以 v0.5.3 标签记录为权威依据。"], ["This page is a retrospective summary of capabilities that shipped in v0.5.3 on 2026-07-08, expressed using the site's current audited fact definitions. Those definitions may evolve and are not a frozen snapshot of that date. GitHub's v0.5.3 release and tagged changelog remain the authoritative historical record for what the release contained and the boundaries documented at the tag.", "本页是对 2026-07-08 随 v0.5.3 交付能力的回顾性摘要，使用网站当前审计的事实定义表述。这些定义可能演进，并非冻结于当日的快照。GitHub 的 v0.5.3 发布记录与标签版更新日志仍是该版本所含内容及标签时边界的权威历史记录。"], ["Historical release", "历史版本"], ["releases", "api-compatibility", "vision-language-models"], [Object.freeze({ type: "release", heading: b("Historical v0.5.3 scope", "历史 v0.5.3 范围"), releaseIds: Object.freeze(["v0-5-3"]) }), sources([], [], ["v0-5-3"])]),
]);
