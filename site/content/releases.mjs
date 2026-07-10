export const releases = Object.freeze([
  Object.freeze({
    id: "v0-5-3",
    version: "0.5.3",
    status: "released",
    releaseDate: "2026-07-08",
    lastVerified: "2026-07-10",
    officialSources: Object.freeze([
      "https://github.com/magicnight/mac-mlx/releases/tag/v0.5.3",
      "https://github.com/magicnight/mac-mlx/blob/v0.5.3/CHANGELOG.md",
    ]),
    shippedFactIds: Object.freeze(["openai-compat", "anthropic-messages", "ollama-selected", "mcp-server", "mcp-client-pool", "embeddings", "rerank-bi-encoder", "tiered-cache", "model-pool", "lora", "vlm-14-families", "deepseek-v32"]),
    limitationFactIds: Object.freeze(["deepseek-v32", "rerank-bi-encoder"]),
    developmentFactIds: Object.freeze(["integrated-tool-routing", "trie-lcp", "continuous-batching", "fixed-prefill-throttle"]),
    plannedFactIds: Object.freeze(["paged-kv", "adaptive-memory-guard", "sampling-expanded"]),
    en: Object.freeze({
      title: "macMLX v0.5.3",
      summary: "The 2026-07-08 release expands compatible APIs, embeddings, model-pool hardening, MCP clients, and the native model stack while keeping post-tag batching work separate.",
      compatibilityNotes: "API compatibility remains endpoint-specific: Anthropic support is Messages-only, Ollama covers five selected endpoints, and /x/models load and unload routes are macMLX extensions.",
      upgradeNotes: "Review the tagged changelog before upgrading. Existing clients should keep their documented endpoint and model assumptions; post-tag batching and prefix reuse are not part of this release.",
    }),
    "zh-Hans": Object.freeze({
      title: "macMLX v0.5.3",
      summary: "2026-07-08 发布版扩展了兼容 API、嵌入、模型池加固、MCP 客户端与原生模型栈，同时明确区分标签后的批处理工作。",
      compatibilityNotes: "API 兼容范围仍按端点界定：Anthropic 仅支持 Messages，Ollama 覆盖五个选定端点，/x/models 的加载与卸载路由属于 macMLX 扩展。",
      upgradeNotes: "升级前请查看标签版更新日志。现有客户端应保留其已记录的端点与模型假设；标签后的批处理与前缀复用不属于本版本。",
    }),
  }),
]);
