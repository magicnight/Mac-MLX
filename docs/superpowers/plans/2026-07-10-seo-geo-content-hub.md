# SEO/GEO Bilingual Content Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate the complete bilingual architecture, API, FAQ, model, comparison, and release route set from one evidence-backed fact model.

**Architecture:** Extend the static foundation with validated product, competitor, FAQ, and release registries. A structured block renderer builds semantic editorial pages from shared facts, while route manifests provide exact English and Chinese counterparts. All visible capability statements originate from the registries and carry status, source, and verification metadata.

**Tech Stack:** Node.js built-ins, ES modules, static HTML/CSS, `node:test`, official product documentation, existing site theme.

---

## Sequence and Dependency

This is plan 2 of 3. Start only after `2026-07-10-seo-geo-foundation.md` passes its completion gate. Finish this plan before `2026-07-10-seo-geo-discovery-validation.md`.

## File Map and Boundaries

- Create `site/content/facts.mjs` — macMLX fact registry and localized capability wording.
- Create `site/content/competitors.mjs` — dated official-source snapshots for Ollama, LM Studio, and oMLX.
- Create `site/content/faqs.mjs` — visible bilingual FAQ answers.
- Create `site/content/releases.mjs` — current release and development boundary.
- Create `site/content/pages.mjs` — 12 content-page definitions with locale metadata and block assignments.
- Create `site/lib/content.mjs` — registry validation and stale-date checks.
- Create `site/lib/blocks.mjs` — semantic renderer for facts, code, FAQs, comparisons, sources, and related links.
- Create `site/templates/article.html` — shared knowledge-page shell.
- Create `site/tests/content.test.mjs` — registry integrity and fact-boundary tests.
- Create `site/tests/blocks.test.mjs` — safe semantic block-rendering tests.
- Create `site/tests/build-content.test.mjs` — route count, counterpart, and visible-content integration tests.
- Modify `site/routes.mjs` — append all content routes.
- Modify `scripts/build-public-site.mjs` — build article pages after home pages.
- Modify `site/assets/css/main.css` — article, breadcrumb, fact, table, code, citation, and mobile-card styles.

### Task 1: Lock the Product Fact Registry and Status Boundaries

**Files:**
- Create: `site/tests/content.test.mjs`
- Create: `site/lib/content.mjs`
- Create: `site/content/facts.mjs`

- [ ] **Step 1: Write failing registry-validation tests**

Create `site/tests/content.test.mjs`:

```js
import assert from "node:assert/strict";
import test from "node:test";
import { facts } from "../content/facts.mjs";
import { validateFacts } from "../lib/content.mjs";

test("all macMLX facts are bilingual, sourced, dated, and status-bound", () => {
  assert.doesNotThrow(() => validateFacts(facts, { today: "2026-07-10", maxAgeDays: 45 }));
  assert.ok(facts.length >= 18);
  assert.ok(facts.some((fact) => fact.status === "released"));
  assert.ok(facts.some((fact) => fact.status === "development"));
  assert.ok(facts.some((fact) => fact.status === "planned"));
});

test("critical claims have the approved status", () => {
  const byId = Object.fromEntries(facts.map((fact) => [fact.id, fact]));
  assert.equal(byId["swift-in-process"].status, "released");
  assert.equal(byId["tiered-kv-cache"].status, "released");
  assert.equal(byId["paged-kv-sharing"].status, "planned");
  assert.equal(byId["adaptive-memory-guard"].status, "planned");
  assert.equal(byId["expanded-sampling-controls"].status, "planned");
});

test("validateFacts rejects stale, missing, or duplicated facts", () => {
  assert.throws(
    () => validateFacts([{ ...facts[0], lastVerified: "2025-01-01" }], { today: "2026-07-10", maxAgeDays: 45 }),
    /stale fact/,
  );
  assert.throws(
    () => validateFacts([{ ...facts[0], zh: undefined }], { today: "2026-07-10", maxAgeDays: 45 }),
    /missing locale content/,
  );
  assert.throws(
    () => validateFacts([facts[0], facts[0]], { today: "2026-07-10", maxAgeDays: 45 }),
    /duplicate fact id/,
  );
});
```

- [ ] **Step 2: Run the test and verify red**

Run:

```bash
node --test site/tests/content.test.mjs
```

Expected: FAIL because the fact and validation modules do not exist.

- [ ] **Step 3: Implement strict registry validation**

Create `site/lib/content.mjs`:

```js
const allowedStatuses = new Set(["released", "development", "planned"]);
const locales = ["en", "zh"];

function dateAgeDays(date, today) {
  return Math.floor((Date.parse(`${today}T00:00:00Z`) - Date.parse(`${date}T00:00:00Z`)) / 86_400_000);
}

export function validateFacts(facts, { today, maxAgeDays }) {
  const ids = new Set();
  for (const fact of facts) {
    if (!fact.id || ids.has(fact.id)) throw new Error(`duplicate fact id: ${fact.id}`);
    ids.add(fact.id);
    if (!allowedStatuses.has(fact.status)) throw new Error(`invalid fact status: ${fact.id}`);
    if (!fact.sinceVersion) throw new Error(`missing sinceVersion: ${fact.id}`);
    if (!/^\d{4}-\d{2}-\d{2}$/.test(fact.lastVerified || "")) throw new Error(`invalid verification date: ${fact.id}`);
    if (dateAgeDays(fact.lastVerified, today) > maxAgeDays) throw new Error(`stale fact: ${fact.id}`);
    if (!Array.isArray(fact.sourceUrls) || fact.sourceUrls.length === 0) throw new Error(`missing source: ${fact.id}`);
    for (const locale of locales) {
      if (!fact[locale]?.title || !fact[locale]?.summary) throw new Error(`missing locale content: ${fact.id}/${locale}`);
    }
  }
}

export function validateDatedRegistry(items, { label, today, maxAgeDays }) {
  const ids = new Set();
  for (const item of items) {
    if (!item.id || ids.has(item.id)) throw new Error(`duplicate ${label} id: ${item.id}`);
    ids.add(item.id);
    if (!item.verifiedVersion) throw new Error(`missing verified version: ${item.id}`);
    if (dateAgeDays(item.lastVerified, today) > maxAgeDays) throw new Error(`stale ${label}: ${item.id}`);
    if (!item.officialSources?.length) throw new Error(`missing official source: ${item.id}`);
    for (const locale of locales) {
      if (!item[locale]) throw new Error(`missing locale content: ${item.id}/${locale}`);
    }
  }
}
```

- [ ] **Step 4: Add the complete initial fact set**

Create `site/content/facts.mjs`:

```js
const readme = "https://github.com/magicnight/mac-mlx/blob/main/README.md";
const changelog = "https://github.com/magicnight/mac-mlx/blob/main/CHANGELOG.md";
const source = (path) => `https://github.com/magicnight/mac-mlx/blob/main/${path}`;

function fact(id, status, sinceVersion, enTitle, enSummary, zhTitle, zhSummary, sourceUrls = [readme]) {
  return {
    id,
    status,
    sinceVersion,
    lastVerified: "2026-07-10",
    sourceUrls,
    en: { title: enTitle, summary: enSummary },
    zh: { title: zhTitle, summary: zhSummary },
  };
}

export const facts = [
  fact("swift-in-process", "released", "0.1.0", "Swift-native in-process inference", "Model loading, generation, caching, and serving run inside the macMLX Swift process through Apple MLX.", "Swift 原生进程内推理", "模型加载、生成、缓存与服务通过 Apple MLX 运行在 macMLX 的 Swift 进程内。", [readme, source("Sources/MacMLXCore")]),
  fact("unified-memory", "released", "0.1.0", "Apple Silicon unified memory", "Weights, activations, and cache share the Mac's unified-memory system across CPU orchestration and integrated-GPU compute.", "Apple 芯片统一内存", "模型权重、激活值和缓存通过统一内存同时服务于 CPU 编排与集成 GPU 计算。"),
  fact("shared-core", "released", "0.1.0", "One core for App, CLI, and API", "The SwiftUI app, macmlx CLI, and compatible HTTP APIs share MacMLXCore behavior and model state.", "应用、CLI 与 API 共用核心", "SwiftUI 应用、macmlx CLI 与兼容 HTTP API 共享 MacMLXCore 的行为和模型状态。"),
  fact("no-python-default", "released", "0.1.0", "No Python runtime in the default path", "The default app and CLI use the Swift engine; optional compatibility engines do not define the normal install path.", "默认路径不需要 Python", "默认应用与 CLI 使用 Swift 引擎；可选兼容引擎不属于常规安装路径。"),
  fact("model-management", "released", "0.5.3", "Model browsing and resumable downloads", "The app browses Hugging Face models, resumes downloads, supports mirrors, and detects updates.", "模型浏览与断点续传", "应用可浏览 Hugging Face 模型、续传下载、使用镜像并检测更新。", [changelog]),
  fact("openai-api", "released", "0.5.3", "OpenAI-compatible APIs", "macMLX exposes compatible chat, completions, embeddings, and model-management surfaces on the local server.", "OpenAI 兼容 API", "macMLX 在本地服务中提供兼容的聊天、补全、嵌入与模型管理接口。", [readme, changelog]),
  fact("anthropic-api", "released", "0.5.3", "Anthropic Messages compatibility", "The local server accepts Anthropic-style /v1/messages requests for supported workflows.", "Anthropic Messages 兼容", "本地服务可为受支持工作流接收 Anthropic 风格的 /v1/messages 请求。", [readme, changelog]),
  fact("ollama-api", "released", "0.5.3", "Ollama-compatible endpoints", "Ollama-compatible endpoints let existing local-model clients connect to macMLX.", "Ollama 兼容端点", "Ollama 兼容端点让现有本地模型客户端能够连接 macMLX。", [readme, changelog]),
  fact("mcp", "released", "0.5.3", "MCP server and client pool", "macMLX includes an MCP server and a client connection pool for tool-oriented local workflows.", "MCP 服务端与客户端池", "macMLX 包含 MCP 服务端与客户端连接池，支持面向工具的本地工作流。", [readme, changelog]),
  fact("embeddings-rerank", "released", "0.5.3", "Embeddings and reranking", "The compatible server supports local embedding and reranking workloads in addition to generation.", "嵌入与重排", "兼容服务除生成外还支持本地嵌入和重排工作负载。", [readme, changelog]),
  fact("tiered-kv-cache", "released", "0.5.3", "Tiered hot and SSD-cold KV cache", "Prompt-cache entries can remain hot in memory or move to a content-addressed SSD cold tier and return later.", "分层热缓存与 SSD 冷缓存", "提示缓存条目可以保留在内存热层，也可以进入内容寻址的 SSD 冷层并在之后恢复。", [changelog]),
  fact("model-pool", "released", "0.5.3", "Memory-aware multi-model pool", "Pool limits, cold swap, idle TTL, memory probes, and eviction keep multiple-model workflows bounded.", "内存感知多模型池", "模型池上限、冷切换、空闲 TTL、内存探针与淘汰机制共同约束多模型工作流。", [changelog]),
  fact("lora", "released", "0.5.3", "LoRA adapter inference", "macMLX can apply supported LoRA adapters within the native inference path.", "LoRA 适配器推理", "macMLX 可在原生推理路径中应用受支持的 LoRA 适配器。", [changelog]),
  fact("vlm-architectures", "released", "0.5.3", "Sixteen vision-language architectures", "The current release supports sixteen vision-language model architecture families in addition to language models.", "十六种视觉语言架构", "当前版本除语言模型外还支持十六种视觉语言模型架构家族。", [changelog]),
  fact("deepseek-v32", "released", "0.5.3", "Pure-Swift DeepSeek V3.2 architecture", "The DeepSeek V3.2 architecture overlay is implemented in Swift and parity tested without claiming universal MoE support.", "纯 Swift DeepSeek V3.2 架构", "DeepSeek V3.2 架构覆盖层由 Swift 实现并经过对齐测试，但不宣称支持所有 MoE 家族。", [changelog]),
  fact("continuous-batching", "development", "0.6.0-dev", "Continuous-batching foundations", "Scheduler, decode-core, cache-infrastructure, and server-routing foundations are merged for the v0.6 development line.", "连续批处理基础", "面向 v0.6 开发线的调度器、解码核心、缓存基础设施与服务路由基础已经合并。", [changelog]),
  fact("paged-kv-sharing", "planned", "0.6.x", "Paged KV sharing and copy-on-write", "Paged allocation, block sharing, longest-prefix reuse, and copy-on-write branching are planned cache-virtualization steps, not v0.5.3 features.", "分页 KV 共享与写时复制", "分页分配、块共享、最长前缀复用与写时复制分支属于规划中的缓存虚拟化步骤，并非 v0.5.3 能力。", [changelog]),
  fact("adaptive-memory-guard", "planned", "0.6.x", "Unified adaptive memory guard", "A feedback layer that coordinates live pressure across cache, model pool, and admission is planned beyond current probes and limits.", "统一自适应内存守卫", "计划在现有探针与上限之外增加反馈层，协调缓存、模型池和准入的实时内存压力。", [changelog]),
  fact("expanded-sampling-controls", "planned", "0.6.x", "Expanded sampling and KV controls", "Top-k, min-p, penalties, per-request seed, and exposed KV-quantization controls remain planned rather than released.", "扩展采样与 KV 控制", "top-k、min-p、惩罚项、逐请求 seed 与开放的 KV 量化控制仍处于规划状态。", [changelog]),
];
```

- [ ] **Step 5: Run tests and commit the fact boundary**

Run `node --test site/tests/content.test.mjs`. Expected: 3 tests PASS.

Commit:

```bash
git add site/content/facts.mjs site/lib/content.mjs site/tests/content.test.mjs
git commit -m "feat(site): make product claims evidence-backed" \
  -m "Constraint: Every reusable claim needs two locales, status, date, and official source" \
  -m "Confidence: high" \
  -m "Scope-risk: moderate" \
  -m "Directive: Never promote planned facts by changing page prose alone" \
  -m "Tested: node --test site/tests/content.test.mjs"
```

### Task 2: Add Dated Competitor, FAQ, and Release Registries

**Files:**
- Create: `site/content/competitors.mjs`
- Create: `site/content/faqs.mjs`
- Create: `site/content/releases.mjs`
- Modify: `site/tests/content.test.mjs`

- [ ] **Step 1: Add failing dated-registry assertions**

Append to `site/tests/content.test.mjs`:

```js
import { competitors } from "../content/competitors.mjs";
import { faqs } from "../content/faqs.mjs";
import { releases } from "../content/releases.mjs";
import { validateDatedRegistry } from "../lib/content.mjs";

test("competitor snapshots are dated and official-source only", () => {
  assert.doesNotThrow(() => validateDatedRegistry(competitors, {
    label: "competitor",
    today: "2026-07-10",
    maxAgeDays: 45,
  }));
  assert.deepEqual(competitors.map((item) => item.id), ["ollama", "lm-studio", "omlx", "swama", "swiftlm"]);
  assert.ok(competitors.every((item) => item.officialSources.every((url) => /^https:\/\//.test(url))));
});

test("FAQ and release entries have complete bilingual answers", () => {
  assert.ok(faqs.length >= 8);
  assert.ok(faqs.every((item) => item.en.question && item.en.answer && item.zh.question && item.zh.answer));
  assert.equal(releases[0].version, "0.5.3");
  assert.equal(releases[0].status, "released");
  assert.ok(releases[0].factIds.length >= 8);
});
```

Run `node --test site/tests/content.test.mjs`. Expected: FAIL for the three missing modules.

- [ ] **Step 2: Create the competitor snapshots**

Create `site/content/competitors.mjs`:

```js
export const competitors = [
  {
    id: "ollama",
    name: "Ollama",
    verifiedVersion: "official documentation snapshot 2026-07-10",
    lastVerified: "2026-07-10",
    officialSources: [
      "https://docs.ollama.com/macos",
      "https://docs.ollama.com/api/introduction",
      "https://docs.ollama.com/modelfile",
    ],
    en: {
      summary: "A cross-platform local-model service centered on a simple app and CLI workflow with Ollama APIs and model packaging.",
      dimensions: {
        platform: "macOS, Windows, and Linux; macOS supports Apple M-series GPU execution and Intel CPU execution.",
        runtime: "A native local service with platform-specific model runners.",
        models: "Ollama model library and Modelfiles, including documented model creation and import workflows.",
        interfaces: "Desktop presence, ollama CLI, native HTTP API, and compatibility surfaces documented by Ollama.",
        bestFit: "Users prioritizing a broadly adopted, simple cross-platform local-model command workflow.",
      },
    },
    zh: {
      summary: "以简洁应用与 CLI 工作流为中心的跨平台本地模型服务，提供 Ollama API 与模型封装方式。",
      dimensions: {
        platform: "支持 macOS、Windows 与 Linux；macOS 支持 Apple M 系列 GPU 执行和 Intel CPU 执行。",
        runtime: "使用平台相关模型运行器的原生本地服务。",
        models: "Ollama 模型库与 Modelfile，包括官方文档中的模型创建和导入流程。",
        interfaces: "桌面入口、ollama CLI、原生 HTTP API 及官方文档中的兼容接口。",
        bestFit: "重视成熟生态、跨平台能力和简洁本地模型命令工作流的用户。",
      },
    },
  },
  {
    id: "lm-studio",
    name: "LM Studio",
    verifiedVersion: "official documentation snapshot 2026-07-10",
    lastVerified: "2026-07-10",
    officialSources: [
      "https://www.lmstudio.ai/docs/app/system-requirements",
      "https://lmstudio.ai/docs/developer/core/server",
      "https://lmstudio.ai/docs/app",
    ],
    en: {
      summary: "A polished cross-platform desktop product with model discovery, chat, CLI, SDK, MCP, and local-server workflows.",
      dimensions: {
        platform: "Apple Silicon macOS, x64/ARM64 Windows, and x64/ARM64 Linux under documented requirements.",
        runtime: "Multiple managed runtimes; llama.cpp across platforms and MLX models on Apple Silicon Macs.",
        models: "GGUF through llama.cpp and MLX models on supported Macs.",
        interfaces: "Desktop GUI, lms CLI, REST API, TypeScript and Python SDKs, OpenAI and Anthropic compatibility, and MCP client features.",
        bestFit: "Users prioritizing a polished model lab, multiple runtimes, and cross-platform desktop workflows.",
      },
    },
    zh: {
      summary: "成熟的跨平台桌面产品，覆盖模型发现、聊天、CLI、SDK、MCP 与本地服务工作流。",
      dimensions: {
        platform: "在官方要求下支持 Apple 芯片 macOS、x64/ARM64 Windows 与 x64/ARM64 Linux。",
        runtime: "管理多种运行时；跨平台使用 llama.cpp，并在 Apple 芯片 Mac 上支持 MLX 模型。",
        models: "通过 llama.cpp 运行 GGUF，并在受支持的 Mac 上运行 MLX 模型。",
        interfaces: "桌面 GUI、lms CLI、REST API、TypeScript/Python SDK、OpenAI/Anthropic 兼容接口与 MCP 客户端能力。",
        bestFit: "重视成熟模型实验室、多运行时和跨平台桌面体验的用户。",
      },
    },
  },
  {
    id: "omlx",
    name: "oMLX",
    verifiedVersion: "official repository snapshot 2026-07-10",
    lastVerified: "2026-07-10",
    officialSources: [
      "https://github.com/jundot/omlx",
      "https://github.com/jundot/omlx/releases",
    ],
    en: {
      summary: "An Apple-Silicon MLX inference server with continuous batching, tiered caching, a web admin surface, and a native SwiftUI macOS app over a Python server core.",
      dimensions: {
        platform: "Apple Silicon macOS.",
        runtime: "Python server core derived from and evolved beyond vllm-mlx, packaged with a native SwiftUI management app.",
        models: "MLX language and vision-model workflows documented by the project.",
        interfaces: "Native macOS app, menu bar, web admin surface, CLI/server workflows, and compatible APIs.",
        bestFit: "Users prioritizing a feature-rich MLX serving backend and advanced batching/cache behavior.",
      },
    },
    zh: {
      summary: "面向 Apple 芯片的 MLX 推理服务，提供连续批处理、分层缓存、Web 管理界面，以及建立在 Python 服务核心之上的原生 SwiftUI macOS 应用。",
      dimensions: {
        platform: "Apple 芯片 macOS。",
        runtime: "由 vllm-mlx 演进而来的 Python 服务核心，并封装原生 SwiftUI 管理应用。",
        models: "项目文档中的 MLX 语言与视觉模型工作流。",
        interfaces: "原生 macOS 应用、菜单栏、Web 管理界面、CLI/服务工作流与兼容 API。",
        bestFit: "重视功能丰富的 MLX 服务后端及高级批处理、缓存能力的用户。",
      },
    },
  },
  {
    id: "swama",
    name: "Swama",
    verifiedVersion: "official repository snapshot 2026-07-10",
    lastVerified: "2026-07-10",
    officialSources: ["https://github.com/Trans-N-ai/swama"],
    en: {
      summary: "A pure-Swift MLX runtime for Apple Silicon with a menu-bar app, CLI, model management, multimodal support, and OpenAI-compatible APIs.",
      dimensions: {
        platform: "Apple Silicon macOS 15 or later under the documented requirements.",
        runtime: "Pure Swift over Apple's MLX Swift bindings.",
        models: "Hugging Face MLX language, vision, embedding, transcription, and experimental speech workflows documented by the project.",
        interfaces: "Native menu-bar app, swama CLI, OpenAI-compatible APIs, and model-management commands.",
        bestFit: "Mac users seeking a broad pure-Swift local AI runtime that includes text, vision, embedding, and audio workflows.",
      },
    },
    zh: {
      summary: "面向 Apple 芯片的纯 Swift MLX 运行时，提供菜单栏应用、CLI、模型管理、多模态支持与 OpenAI 兼容 API。",
      dimensions: {
        platform: "按照官方要求支持 Apple 芯片 macOS 15 或更高版本。",
        runtime: "基于 Apple MLX Swift 绑定的纯 Swift 运行时。",
        models: "项目文档中的 Hugging Face MLX 语言、视觉、嵌入、语音识别与实验性语音工作流。",
        interfaces: "原生菜单栏应用、swama CLI、OpenAI 兼容 API 与模型管理命令。",
        bestFit: "希望通过纯 Swift 运行时覆盖文本、视觉、嵌入与音频工作流的 Mac 用户。",
      },
    },
  },
  {
    id: "swiftlm",
    name: "SwiftLM",
    verifiedVersion: "official repository snapshot 2026-07-10",
    lastVerified: "2026-07-10",
    officialSources: ["https://github.com/SharpAI/SwiftLM"],
    en: {
      summary: "A native Swift MLX inference server focused on Apple Silicon, OpenAI-compatible serving, and very large MoE models through SSD-oriented techniques.",
      dimensions: {
        platform: "Apple Silicon macOS, with additional iOS work documented by the project.",
        runtime: "Native Swift and MLX with no Python server in the documented core path.",
        models: "MLX language models with emphasis on large mixture-of-experts inference.",
        interfaces: "OpenAI-compatible server and project-specific native app surfaces.",
        bestFit: "Users researching native Swift serving and very large MoE inference on high-memory Apple Silicon systems.",
      },
    },
    zh: {
      summary: "面向 Apple 芯片的原生 Swift MLX 推理服务，重点覆盖 OpenAI 兼容服务与通过 SSD 技术运行超大 MoE 模型。",
      dimensions: {
        platform: "Apple 芯片 macOS，并包含项目文档中的 iOS 工作。",
        runtime: "原生 Swift 与 MLX，文档中的核心路径不依赖 Python 服务。",
        models: "MLX 语言模型，重点关注超大混合专家模型推理。",
        interfaces: "OpenAI 兼容服务与项目特定的原生应用入口。",
        bestFit: "研究原生 Swift 服务及高内存 Apple 芯片上超大 MoE 推理的用户。",
      },
    },
  },
];
```

- [ ] **Step 3: Create the real FAQ and release entries**

Create `site/content/faqs.mjs`:

```js
function faq(id, enQuestion, enAnswer, zhQuestion, zhAnswer, factIds) {
  return { id, factIds, en: { question: enQuestion, answer: enAnswer }, zh: { question: zhQuestion, answer: zhAnswer } };
}

export const faqs = [
  faq("requirements", "What Mac does macMLX require?", "macMLX requires macOS 14 or later on Apple Silicon. Intel Macs, Windows, and Linux are outside the native app's supported platform.", "macMLX 需要什么 Mac？", "macMLX 需要运行 macOS 14 或更高版本的 Apple 芯片 Mac。原生应用不支持 Intel Mac、Windows 或 Linux。", ["unified-memory"]),
  faq("python", "Does macMLX require Python?", "No for the default app, CLI, and Swift-native inference path. Optional compatibility engines may use other runtimes, but they are not the default installation.", "macMLX 需要 Python 吗？", "默认应用、CLI 与 Swift 原生推理路径不需要 Python。可选兼容引擎可能使用其他运行时，但不属于默认安装。", ["no-python-default"]),
  faq("privacy", "Does inference leave the Mac?", "Inference and the local API stay on the Mac by default. Downloading models and checking remote repositories still require network access.", "推理数据会离开 Mac 吗？", "推理和本地 API 默认留在 Mac 上。下载模型及检查远程仓库仍然需要网络连接。", ["swift-in-process"]),
  faq("models", "Which model format should I download?", "Choose an MLX-format model compatible with the architecture and memory available on your Mac. The model guides explain task, quantization, and vision tradeoffs.", "应该下载哪种模型格式？", "请选择与模型架构和 Mac 可用内存匹配的 MLX 格式模型。模型指南会说明任务、量化与视觉能力的取舍。", ["model-management"]),
  faq("memory", "How much memory does a model need?", "Memory depends on parameter count, quantization, context length, cache size, and concurrent models. Leave headroom for macOS and begin with a smaller quantized model.", "模型需要多少内存？", "内存占用取决于参数量、量化、上下文长度、缓存大小和并发模型数量。请为 macOS 保留余量，并优先从较小的量化模型开始。", ["model-pool", "tiered-kv-cache"]),
  faq("api", "Can existing OpenAI clients connect?", "Yes. Point a compatible client at the local macMLX server and use the documented OpenAI-compatible endpoints. Endpoint support is listed on the API compatibility page.", "现有 OpenAI 客户端可以连接吗？", "可以。将兼容客户端指向 macMLX 本地服务，并使用文档中的 OpenAI 兼容端点。具体支持范围见 API 兼容性页面。", ["openai-api"]),
  faq("gatekeeper", "What if Gatekeeper blocks the unsigned app?", "Use the project's installation documentation for the current Gatekeeper process. Do not disable system-wide security protections solely to run macMLX.", "Gatekeeper 阻止未签名应用怎么办？", "请按照项目安装文档中的当前 Gatekeeper 操作处理。不要仅为了运行 macMLX 而全局关闭系统安全保护。", []),
  faq("vision", "Can macMLX run vision-language models?", "Yes. v0.5.3 supports sixteen documented vision-language architecture families, subject to model-specific compatibility and memory limits.", "macMLX 可以运行视觉语言模型吗？", "可以。v0.5.3 支持十六种已记录的视觉语言架构家族，但仍受具体模型兼容性与内存限制。", ["vlm-architectures"]),
];
```

Create `site/content/releases.mjs`:

```js
export const releases = [
  {
    id: "v0-5-3",
    version: "0.5.3",
    verifiedVersion: "v0.5.3",
    status: "released",
    releaseDate: "2026-07-08",
    lastVerified: "2026-07-10",
    officialSources: [
      "https://github.com/magicnight/mac-mlx/releases",
      "https://github.com/magicnight/mac-mlx/blob/main/CHANGELOG.md",
    ],
    factIds: ["model-management", "openai-api", "anthropic-api", "ollama-api", "mcp", "embeddings-rerank", "tiered-kv-cache", "model-pool", "lora", "vlm-architectures", "deepseek-v32"],
    developmentFactIds: ["continuous-batching", "paged-kv-sharing", "adaptive-memory-guard", "expanded-sampling-controls"],
    en: {
      title: "macMLX v0.5.3",
      summary: "A local inference release focused on model management, compatible APIs, caching, multi-model workflows, adapters, and broad vision-model support.",
      limitation: "v0.6 batching and cache-virtualization work is development scope, not part of v0.5.3.",
    },
    zh: {
      title: "macMLX v0.5.3",
      summary: "这一版本聚焦模型管理、兼容 API、缓存、多模型工作流、适配器与广泛的视觉模型支持。",
      limitation: "v0.6 的批处理与缓存虚拟化工作属于开发范围，不包含在 v0.5.3 中。",
    },
  },
];
```

- [ ] **Step 4: Run registry tests and commit**

Run `node --test site/tests/content.test.mjs`. Expected: 5 tests PASS.

Commit the three registries and tests with:

```bash
git add site/content/competitors.mjs site/content/faqs.mjs site/content/releases.mjs site/tests/content.test.mjs
git commit -m "feat(site): date comparisons and support answers" \
  -m "Constraint: Competitor claims use official sources and dated snapshots" \
  -m "Rejected: Winner scores | not verifiable across changing versions" \
  -m "Confidence: medium" \
  -m "Scope-risk: moderate" \
  -m "Directive: Reverify competitor snapshots with each macMLX release" \
  -m "Tested: node --test site/tests/content.test.mjs"
```

### Task 3: Implement the Semantic Content-Block Renderer

**Files:**
- Create: `site/tests/blocks.test.mjs`
- Create: `site/lib/blocks.mjs`
- Create: `site/templates/article.html`

- [ ] **Step 1: Write failing renderer tests**

Create `site/tests/blocks.test.mjs`:

```js
import assert from "node:assert/strict";
import test from "node:test";
import { renderBlocks } from "../lib/blocks.mjs";

const context = {
  copyLocale: "en",
  factsById: {
    core: {
      id: "core",
      status: "released",
      sinceVersion: "0.5.3",
      en: { title: "Swift core", summary: "One process." },
      zh: { title: "Swift 核心", summary: "单一进程。" },
    },
  },
  faqsById: {
    python: {
      en: { question: "Does it need Python?", answer: "Not on the default path." },
      zh: { question: "需要 Python 吗？", answer: "默认路径不需要。" },
    },
  },
  competitorsById: {
    omlx: {
      name: "oMLX",
      lastVerified: "2026-07-10",
      verifiedVersion: "official repository snapshot 2026-07-10",
      en: { summary: "MLX server.", dimensions: { runtime: "Python server core." } },
      zh: { summary: "MLX 服务。", dimensions: { runtime: "Python 服务核心。" } },
    },
  },
};

test("renderBlocks emits semantic facts, code, FAQ, comparisons, sources, and related links", () => {
  const html = renderBlocks([
    { type: "facts", heading: "Facts", factIds: ["core"] },
    { type: "code", heading: "Example", intro: "Call the API.", code: "curl http://localhost:8000/v1" },
    { type: "table", heading: "Matrix", caption: "Compatibility matrix", headers: ["Surface", "Status"], rows: [["Chat", "Released"]] },
    { type: "faq", heading: "FAQ", faqIds: ["python"] },
    {
      type: "comparison",
      heading: "Runtime",
      competitorId: "omlx",
      verifiedLabel: "Last verified:",
      rows: [{ key: "runtime", label: "Core runtime", macmlx: "Swift in-process." }],
    },
    { type: "sources", heading: "Sources", sources: [{ label: "README", url: "https://github.com/magicnight/mac-mlx" }] },
    { type: "related", heading: "Related", ariaLabel: "Related pages", links: [{ href: "/architecture/", title: "Architecture", description: "Read how it works." }] },
  ], context);
  assert.match(html, /class="fact-card"/);
  assert.match(html, /<pre class="article-code"><code>/);
  assert.match(html, /<table class="article-table"><caption>Compatibility matrix<\/caption>/);
  assert.match(html, /<details><summary>/);
  assert.match(html, /<table class="comparison-table"><caption>/);
  assert.match(html, /<th scope="row">Core runtime<\/th>/);
  assert.match(html, /class="sources"/);
  assert.match(html, /class="related-pages"/);
});

test("renderBlocks escapes registry values and rejects unknown blocks", () => {
  const html = renderBlocks([{ type: "paragraph", text: `<script>alert(1)</script>` }], context);
  assert.doesNotMatch(html, /<script>/);
  assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.throws(() => renderBlocks([{ type: "unknown" }], context), /unsupported block type/);
});
```

Run `node --test site/tests/blocks.test.mjs`. Expected: FAIL because `site/lib/blocks.mjs` is missing.

- [ ] **Step 2: Implement the complete supported block set**

Create `site/lib/blocks.mjs` with these exported block types and no others:

```js
import { escapeHTML } from "./localize.mjs";

const statusLabels = {
  en: { released: "Released", development: "In development", planned: "Planned" },
  zh: { released: "已发布", development: "开发中", planned: "规划中" },
};

function renderFactList(block, context) {
  const facts = block.factIds.map((id) => context.factsById[id]);
  return `<section class="content-section"><h2>${escapeHTML(block.heading)}</h2><div class="fact-cards">${facts.map((fact) => {
    const copy = fact[context.copyLocale];
    return `<article class="fact-card" data-status="${fact.status}"><div class="fact-card__meta"><span>${escapeHTML(statusLabels[context.copyLocale][fact.status])}</span><span>${escapeHTML(fact.sinceVersion)}</span></div><h3>${escapeHTML(copy.title)}</h3><p>${escapeHTML(copy.summary)}</p></article>`;
  }).join("")}</div></section>`;
}

function renderCode(block) {
  return `<section class="content-section"><h2>${escapeHTML(block.heading)}</h2><p>${escapeHTML(block.intro)}</p><pre class="article-code"><code>${escapeHTML(block.code)}</code></pre></section>`;
}

function renderTable(block) {
  return `<section class="content-section"><h2>${escapeHTML(block.heading)}</h2><div class="comparison-wrap"><table class="article-table"><caption>${escapeHTML(block.caption)}</caption><thead><tr>${block.headers.map((header) => `<th scope="col">${escapeHTML(header)}</th>`).join("")}</tr></thead><tbody>${block.rows.map((row) => `<tr>${row.map((cell, index) => index === 0 ? `<th scope="row">${escapeHTML(cell)}</th>` : `<td>${escapeHTML(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div></section>`;
}

function renderFAQ(block, context) {
  return `<section class="content-section"><h2>${escapeHTML(block.heading)}</h2><div class="faq-list">${block.faqIds.map((id) => {
    const item = context.faqsById[id][context.copyLocale];
    return `<details><summary>${escapeHTML(item.question)}</summary><p>${escapeHTML(item.answer)}</p></details>`;
  }).join("")}</div></section>`;
}

function renderComparison(block, context) {
  const competitor = context.competitorsById[block.competitorId];
  const competitorCopy = competitor[context.copyLocale];
  const labels = context.copyLocale === "en"
    ? { caption: `macMLX and ${competitor.name} factual comparison`, dimension: "Dimension", macmlx: "macMLX", other: competitor.name }
    : { caption: `macMLX 与 ${competitor.name} 事实对比`, dimension: "维度", macmlx: "macMLX", other: competitor.name };
  return `<section class="content-section"><h2>${escapeHTML(block.heading)}</h2><p>${escapeHTML(competitorCopy.summary)}</p><div class="comparison-wrap"><table class="comparison-table"><caption>${escapeHTML(labels.caption)}</caption><thead><tr><th>${escapeHTML(labels.dimension)}</th><th>${labels.macmlx}</th><th>${escapeHTML(labels.other)}</th></tr></thead><tbody>${block.rows.map((row) => `<tr><th scope="row">${escapeHTML(row.label)}</th><td>${escapeHTML(row.macmlx)}</td><td>${escapeHTML(competitorCopy.dimensions[row.key])}</td></tr>`).join("")}</tbody></table></div><p class="verified-note">${escapeHTML(block.verifiedLabel)} ${escapeHTML(competitor.lastVerified)} · ${escapeHTML(competitor.verifiedVersion)}</p></section>`;
}

function renderSources(block) {
  return `<section class="content-section sources"><h2>${escapeHTML(block.heading)}</h2><ol>${block.sources.map((source) => `<li><a href="${escapeHTML(source.url)}">${escapeHTML(source.label)}</a></li>`).join("")}</ol></section>`;
}

function renderRelated(block) {
  return `<nav class="related-pages" aria-label="${escapeHTML(block.ariaLabel)}"><h2>${escapeHTML(block.heading)}</h2><div>${block.links.map((link) => `<a href="${escapeHTML(link.href)}"><strong>${escapeHTML(link.title)}</strong><span>${escapeHTML(link.description)}</span></a>`).join("")}</div></nav>`;
}

export function renderBlocks(blocks, context) {
  return blocks.map((block) => {
    if (block.type === "paragraph") return `<p class="article-paragraph">${escapeHTML(block.text)}</p>`;
    if (block.type === "facts") return renderFactList(block, context);
    if (block.type === "code") return renderCode(block);
    if (block.type === "table") return renderTable(block);
    if (block.type === "faq") return renderFAQ(block, context);
    if (block.type === "comparison") return renderComparison(block, context);
    if (block.type === "sources") return renderSources(block);
    if (block.type === "related") return renderRelated(block);
    throw new Error(`unsupported block type: ${block.type}`);
  }).join("\n");
}
```

- [ ] **Step 3: Create the shared article template**

Create `site/templates/article.html` with complete document semantics and these required raw insertion points:

```html
<!doctype html>
<html lang="{{htmlLang}}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {{{head}}}
  <meta name="theme-color" content="#f3f1ea" media="(prefers-color-scheme: light)">
  <meta name="theme-color" content="#111311" media="(prefers-color-scheme: dark)">
  <meta name="color-scheme" content="light dark">
  <link rel="stylesheet" href="/assets/css/main.css?v=2026071005">
</head>
<body data-lang="{{bodyLanguage}}">
  <a class="skip-link" href="#main">{{skipLabel}}</a>
  {{{siteHeader}}}
  <main id="main" class="article-main">
    <div class="page-shell article-shell">
      {{{breadcrumbs}}}
      <header class="article-hero">
        <p class="article-eyebrow">{{eyebrow}}</p>
        <h1>{{title}}</h1>
        <p class="article-answer">{{directAnswer}}</p>
        <p class="verified-note">{{verifiedLabel}} {{lastVerified}}</p>
      </header>
      <article class="article-body">
        {{{content}}}
      </article>
    </div>
  </main>
  {{{siteFooter}}}
  <script src="/assets/js/main.js?v=2026071004" defer></script>
</body>
</html>
```

All normal tokens are escaped. Only `head`, `siteHeader`, `breadcrumbs`, `content`, and `siteFooter` are trusted raw tokens built from repository-owned functions.

- [ ] **Step 4: Run renderer tests and commit**

Run `node --test site/tests/blocks.test.mjs`. Expected: all tests PASS.

Commit:

```bash
git add site/lib/blocks.mjs site/templates/article.html site/tests/blocks.test.mjs
git commit -m "feat(site): render citation-ready technical articles" \
  -m "Constraint: Registry content is escaped and only known block types render" \
  -m "Confidence: high" \
  -m "Scope-risk: moderate" \
  -m "Tested: node --test site/tests/blocks.test.mjs"
```

### Task 4: Define the Complete Route and Page Catalog

**Files:**
- Create: `site/content/pages.mjs`
- Modify: `site/routes.mjs`
- Create: `site/tests/build-content.test.mjs`

- [ ] **Step 1: Write the failing route-set integration contract**

Create `site/tests/build-content.test.mjs`:

```js
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { buildSite } from "../../scripts/build-public-site.mjs";
import { routes } from "../routes.mjs";

const expectedIds = [
  "home", "architecture", "api-compatibility", "faq", "models",
  "choosing-a-model", "vision-language-models", "compare", "compare-ollama",
  "compare-lm-studio", "compare-omlx", "releases", "release-v0-5-3",
];

test("route catalog contains the complete approved hub", () => {
  assert.deepEqual(routes.map((route) => route.id), expectedIds);
  assert.equal(new Set(routes.flatMap((route) => Object.values(route.paths))).size, 26);
});

test("build emits every article counterpart with visible verification", async () => {
  await buildSite();
  for (const route of routes.filter((item) => item.kind === "article")) {
    for (const [locale, routePath] of Object.entries(route.paths)) {
      const output = routePath === "/" ? "index.html" : `${routePath.slice(1)}index.html`;
      const html = await readFile(new URL(`../../public/${output}`, import.meta.url), "utf8");
      assert.match(html, /<article class="article-body">/);
      assert.match(html, /2026-07-10/);
      assert.match(html, locale === "en" ? /<html lang="en">/ : /<html lang="zh-CN">/);
    }
  }
});
```

Run `node --test site/tests/build-content.test.mjs`. Expected: FAIL because only `home` exists.

- [ ] **Step 2: Create the page catalog with exact search intent**

Create `site/content/pages.mjs`. Use a `copy(en, zh)` helper and define these 12 pages exactly:

```js
const copy = (en, zh) => ({ en, "zh-Hans": zh });
const page = (id, path, eyebrow, title, description, directAnswer, configuration = {}) => ({
  id,
  kind: "article",
  template: "article",
  paths: { en: path, "zh-Hans": `/zh${path}` },
  eyebrow: copy(eyebrow[0], eyebrow[1]),
  title: copy(title[0], title[1]),
  directAnswer: copy(directAnswer[0], directAnswer[1]),
  metadata: {
    en: { title: `${title[0]} — macMLX`, description: description[0], socialDescription: description[0] },
    "zh-Hans": { title: `${title[1]} — macMLX`, description: description[1], socialDescription: description[1] },
  },
  lastVerified: "2026-07-10",
  ...configuration,
});

export const pages = [
  page("architecture", "/architecture/", ["Architecture", "架构原理"], ["How macMLX runs LLMs on Apple Silicon", "macMLX 如何在 Apple 芯片上运行大模型"], ["Understand the Swift in-process MLX engine, unified memory, shared App/CLI/API core, caching, and current development boundaries.", "了解 Swift 进程内 MLX 引擎、统一内存、应用/CLI/API 共享核心、缓存机制与当前开发边界。"], ["macMLX loads, generates, caches, and serves models inside one Swift process through Apple MLX. The app, CLI, and APIs share that core instead of coordinating a default Python sidecar.", "macMLX 通过 Apple MLX 在同一个 Swift 进程内完成模型加载、生成、缓存与服务。应用、CLI 和 API 共享这一核心，而不是协调默认的 Python 旁路服务。"], { factIds: ["swift-in-process", "unified-memory", "shared-core", "no-python-default", "tiered-kv-cache", "continuous-batching", "paged-kv-sharing"] }),
  page("api-compatibility", "/api-compatibility/", ["API compatibility", "API 兼容性"], ["Local APIs for existing AI clients", "面向现有 AI 客户端的本地 API"], ["See which OpenAI, Anthropic, Ollama, MCP, embedding, and reranking workflows macMLX supports locally.", "查看 macMLX 在本地支持哪些 OpenAI、Anthropic、Ollama、MCP、嵌入和重排工作流。"], ["macMLX exposes several compatibility surfaces from the same local Swift server. Compatibility is endpoint-specific, so clients should use the documented matrix instead of assuming every provider extension is identical.", "macMLX 通过同一个本地 Swift 服务提供多种兼容接口。兼容性取决于具体端点，客户端应以文档矩阵为准，而不是假设所有提供方扩展完全一致。"], { factIds: ["openai-api", "anthropic-api", "ollama-api", "mcp", "embeddings-rerank"], code: "curl http://localhost:8000/v1/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"model\":\"Qwen3-8B-4bit\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'" }),
  page("faq", "/faq/", ["FAQ", "常见问题"], ["macMLX questions, answered", "macMLX 常见问题解答"], ["Requirements, Python, privacy, models, memory, APIs, Gatekeeper, and vision support explained directly.", "直接说明系统要求、Python、隐私、模型、内存、API、Gatekeeper 与视觉支持。"], ["The answers below describe the current supported release and link to deeper technical pages where behavior depends on model, memory, or client compatibility.", "以下答案描述当前受支持版本；涉及模型、内存或客户端兼容性的行为会链接到更深入的技术页面。"], { faqIds: ["requirements", "python", "privacy", "models", "memory", "api", "gatekeeper", "vision"] }),
  page("models", "/models/", ["Model guides", "模型指南"], ["Choose an MLX model for your Mac", "为你的 Mac 选择 MLX 模型"], ["Start with memory, task, context, quantization, and vision requirements before downloading an MLX model.", "下载 MLX 模型前，先根据内存、任务、上下文、量化和视觉需求做选择。"], ["There is no single best local model for every Mac. The practical choice is the smallest model and quantization that meets the task while leaving enough unified-memory headroom for macOS, context, and cache.", "不存在适合所有 Mac 的唯一最佳本地模型。实际选择应是在满足任务的前提下，选用最小的模型与量化，并为 macOS、上下文和缓存保留足够统一内存余量。"], { factIds: ["model-management", "unified-memory", "model-pool", "vlm-architectures"] }),
  page("choosing-a-model", "/models/choosing-a-model/", ["Model selection", "模型选择"], ["How to size a local model", "如何估算本地模型规模"], ["A practical decision guide for parameter count, quantization, context length, concurrency, and memory headroom.", "面向参数量、量化、上下文长度、并发与内存余量的实用决策指南。"], ["Estimate the model weights first, then add context-cache, runtime, concurrent-model, and macOS headroom. If the result is close to physical memory, choose a smaller or more strongly quantized model.", "先估算模型权重，再加入上下文缓存、运行时、并发模型和 macOS 余量。如果总量接近物理内存，应选择更小或量化程度更高的模型。"], { factIds: ["unified-memory", "tiered-kv-cache", "model-pool"] }),
  page("vision-language-models", "/models/vision-language-models/", ["Vision models", "视觉模型"], ["Run vision-language models locally", "在本地运行视觉语言模型"], ["Understand architecture compatibility, image inputs, memory cost, and the v0.5.3 vision-model boundary.", "了解架构兼容性、图片输入、内存成本与 v0.5.3 的视觉模型边界。"], ["macMLX v0.5.3 supports sixteen documented vision-language architecture families, but a repository being labeled MLX does not guarantee that its architecture and image processor are supported.", "macMLX v0.5.3 支持十六种已记录的视觉语言架构家族，但仓库标注为 MLX 并不代表其架构与图像处理器一定受支持。"], { factIds: ["vlm-architectures", "model-management", "unified-memory"] }),
  page("compare", "/compare/", ["Comparisons", "产品对比"], ["Compare local LLM tools for Mac", "对比 Mac 本地大模型工具"], ["A dated, source-linked comparison of macMLX, Ollama, LM Studio, oMLX, Swama, and SwiftLM by technical route and target user.", "按技术路线和目标用户，对 macMLX、Ollama、LM Studio、oMLX、Swama 与 SwiftLM 进行带日期和来源的对比。"], ["The right tool depends on whether you prioritize a Swift-native MLX core, cross-platform model support, polished desktop workflows, or advanced serving features. This overview separates those routes without declaring a universal winner.", "合适的工具取决于你更重视 Swift 原生 MLX 核心、跨平台模型支持、成熟桌面工作流，还是高级服务能力。本页区分这些路线，但不宣布绝对赢家。"], { factIds: ["swift-in-process", "shared-core", "no-python-default"], competitorIds: ["ollama", "lm-studio", "omlx", "swama", "swiftlm"] }),
  page("compare-ollama", "/compare/ollama/", ["Comparison", "产品对比"], ["macMLX vs Ollama", "macMLX 与 Ollama 对比"], ["Compare macMLX and Ollama by platform, runtime, model workflow, interfaces, and best-fit user.", "按平台、运行时、模型工作流、接口和适用用户对比 macMLX 与 Ollama。"], ["macMLX concentrates on an in-process Swift MLX stack for Apple Silicon, while Ollama emphasizes a simple, broadly adopted cross-platform service and CLI workflow. Both are local-first, but their runtime and ecosystem choices differ.", "macMLX 聚焦 Apple 芯片上的 Swift 进程内 MLX 栈；Ollama 更强调简洁、成熟且跨平台的服务与 CLI 工作流。两者都以本地为先，但运行时与生态选择不同。"], { competitorId: "ollama" }),
  page("compare-lm-studio", "/compare/lm-studio/", ["Comparison", "产品对比"], ["macMLX vs LM Studio", "macMLX 与 LM Studio 对比"], ["Compare macMLX and LM Studio by operating systems, runtimes, model formats, interfaces, and product focus.", "按操作系统、运行时、模型格式、接口与产品重点对比 macMLX 和 LM Studio。"], ["macMLX is a focused open-source Apple-Silicon stack with one Swift MLX core. LM Studio is a broader cross-platform desktop model lab with multiple runtimes, SDKs, and polished discovery workflows.", "macMLX 是聚焦 Apple 芯片、以单一 Swift MLX 核心为中心的开源栈；LM Studio 是覆盖多运行时、SDK 与成熟模型发现工作流的跨平台桌面模型实验室。"], { competitorId: "lm-studio" }),
  page("compare-omlx", "/compare/omlx/", ["Comparison", "产品对比"], ["macMLX vs oMLX", "macMLX 与 oMLX 对比"], ["Compare two Apple-Silicon MLX tools by engine runtime, desktop surface, serving focus, and cache architecture.", "按引擎运行时、桌面入口、服务重点与缓存架构对比两款 Apple 芯片 MLX 工具。"], ["Both projects target MLX inference on Apple Silicon and offer native macOS management surfaces. macMLX differentiates through a Swift in-process inference core; oMLX differentiates through a feature-rich Python serving core with advanced batching and caching.", "两者都面向 Apple 芯片上的 MLX 推理，并提供原生 macOS 管理入口。macMLX 的差异在于 Swift 进程内推理核心；oMLX 的差异在于功能丰富、具备高级批处理和缓存能力的 Python 服务核心。"], { competitorId: "omlx" }),
  page("releases", "/releases/", ["Releases", "版本"], ["macMLX release notes", "macMLX 版本说明"], ["Search-friendly release summaries with links to authoritative GitHub release assets and changelogs.", "便于检索的版本摘要，并链接到权威 GitHub 发布资源与更新日志。"], ["Release pages summarize what shipped, what remains limited, and what moved into development. GitHub Releases and the changelog remain the authoritative distribution records.", "版本页面概括已交付能力、当前限制与进入开发阶段的工作。GitHub Releases 与更新日志仍是权威发布记录。"], { releaseIds: ["v0-5-3"] }),
  page("release-v0-5-3", "/releases/v0-5-3/", ["Release", "版本"], ["macMLX v0.5.3 release", "macMLX v0.5.3 版本"], ["What shipped in macMLX v0.5.3, current limitations, and the boundary with v0.6 development work.", "说明 macMLX v0.5.3 的已交付内容、当前限制，以及与 v0.6 开发工作的边界。"], ["macMLX v0.5.3 expands the local inference stack across model management, compatibility APIs, caching, multi-model operation, adapters, and vision architectures while keeping v0.6 batching work explicitly separate.", "macMLX v0.5.3 扩展了模型管理、兼容 API、缓存、多模型运行、适配器与视觉架构能力，同时明确将 v0.6 批处理工作保持在当前版本之外。"], { releaseId: "v0-5-3" }),
];

const pageById = Object.fromEntries(pages.map((item) => [item.id, item]));

pageById["api-compatibility"].table = copy(
  {
    heading: "Compatibility matrix",
    caption: "Released v0.5.3 local interfaces",
    headers: ["Surface", "Route or command", "Boundary"],
    rows: [
      ["OpenAI chat", "/v1/chat/completions", "Released"],
      ["OpenAI completions", "/v1/completions", "Released"],
      ["Embeddings", "/v1/embeddings", "Released"],
      ["Anthropic messages", "/v1/messages", "Released"],
      ["Ollama clients", "Ollama-compatible routes", "Endpoint-specific"],
      ["MCP", "macmlx mcp serve", "Server and client-pool workflows"],
    ],
  },
  {
    heading: "兼容性矩阵",
    caption: "v0.5.3 已发布的本地接口",
    headers: ["接口", "路径或命令", "边界"],
    rows: [
      ["OpenAI 聊天", "/v1/chat/completions", "已发布"],
      ["OpenAI 补全", "/v1/completions", "已发布"],
      ["嵌入", "/v1/embeddings", "已发布"],
      ["Anthropic messages", "/v1/messages", "已发布"],
      ["Ollama 客户端", "Ollama 兼容路由", "依具体端点而定"],
      ["MCP", "macmlx mcp serve", "服务端与客户端池工作流"],
    ],
  },
);

pageById["choosing-a-model"].table = copy(
  {
    heading: "Sizing decisions",
    caption: "Inputs that change local memory pressure",
    headers: ["Factor", "Start with", "Why"],
    rows: [
      ["Parameter count", "The smallest model that meets the task", "Weights dominate the baseline footprint"],
      ["Quantization", "A supported 4-bit or 8-bit build", "Lower precision reduces weight memory"],
      ["Context length", "The shortest useful context", "Longer context grows KV cache"],
      ["Concurrency", "One loaded model", "Pools and concurrent requests add pressure"],
      ["Headroom", "Leave memory for macOS", "Near-capacity loads are less stable"],
    ],
  },
  {
    heading: "规模选择",
    caption: "影响本地内存压力的因素",
    headers: ["因素", "起步建议", "原因"],
    rows: [
      ["参数量", "满足任务的最小模型", "权重决定基础占用"],
      ["量化", "受支持的 4-bit 或 8-bit 版本", "更低精度减少权重内存"],
      ["上下文长度", "满足需求的最短上下文", "更长上下文会增大 KV 缓存"],
      ["并发", "先加载一个模型", "模型池与并发请求会增加压力"],
      ["余量", "为 macOS 保留内存", "接近容量上限时稳定性更差"],
    ],
  },
);

pageById["vision-language-models"].table = copy(
  {
    heading: "Before downloading",
    caption: "Vision-model compatibility checks",
    headers: ["Check", "Question", "Impact"],
    rows: [
      ["Architecture", "Is the exact model family supported?", "Unsupported processors cannot load correctly"],
      ["Image processor", "Does the repository include compatible preprocessing?", "Vision inputs depend on model-specific transforms"],
      ["Memory", "Do weights, image tokens, and cache fit with headroom?", "Vision workloads add tokens and activations"],
      ["Task", "Is image understanding actually required?", "Text-only models are usually lighter"],
    ],
  },
  {
    heading: "下载前检查",
    caption: "视觉模型兼容性检查",
    headers: ["检查项", "问题", "影响"],
    rows: [
      ["架构", "是否支持确切的模型家族？", "不受支持的处理器无法正确加载"],
      ["图像处理器", "仓库是否包含兼容预处理？", "视觉输入依赖模型特定变换"],
      ["内存", "权重、图像词元与缓存是否留有余量？", "视觉工作负载会增加词元与激活值"],
      ["任务", "是否确实需要图像理解？", "纯文本模型通常更轻量"],
    ],
  },
);
```

- [ ] **Step 3: Append pages to the route manifest**

Replace `site/routes.mjs` with:

```js
import { pages } from "./content/pages.mjs";

const homeRoute = {
  id: "home",
  kind: "home",
  template: "home",
  paths: { en: "/", "zh-Hans": "/zh/" },
  metadata: {
    en: {
      title: "macMLX — Native Swift inference for Apple Silicon",
      description: "Run local language and vision models through a native SwiftUI app, CLI, and compatible API, all powered by one Swift in-process MLX engine.",
      socialDescription: "A native SwiftUI app, CLI, and compatible API over one in-process MLX engine.",
    },
    "zh-Hans": {
      title: "macMLX — Apple 芯片上的原生 Swift 推理",
      description: "通过原生 SwiftUI 应用、CLI 与兼容 API 在 Mac 上运行本地语言和视觉模型，共用一个 Swift 进程内 MLX 引擎。",
      socialDescription: "原生 SwiftUI 应用、CLI 与兼容 API，共用一个 Swift 进程内 MLX 引擎。",
    },
  },
};

export const routes = [homeRoute, ...pages];
```

- [ ] **Step 4: Run the route-set test and keep the build portion red**

Run `node --test site/tests/build-content.test.mjs`.

Expected: the route-catalog assertion passes; the output assertion fails because article generation is not implemented.

- [ ] **Step 5: Commit the approved information architecture**

```bash
git add site/content/pages.mjs site/routes.mjs site/tests/build-content.test.mjs
git commit -m "feat(site): define the complete bilingual knowledge hub" \
  -m "Constraint: Every English route has one exact Simplified Chinese counterpart" \
  -m "Confidence: high" \
  -m "Scope-risk: moderate" \
  -m "Tested: route catalog portion of site/tests/build-content.test.mjs"
```

### Task 5: Generate the Article Pages and Shared Editorial Navigation

**Files:**
- Modify: `scripts/build-public-site.mjs`
- Modify: `site/assets/css/main.css`
- Modify: `site/tests/build-content.test.mjs`
- Generate: all approved `public/**/index.html` article routes

- [ ] **Step 1: Add failing visible-content assertions**

Extend `site/tests/build-content.test.mjs` after reading generated HTML:

```js
      assert.match(html, /class="article-answer"/);
      assert.match(html, /class="site-header"/);
      assert.match(html, /class="site-footer"/);
      assert.match(html, /class="breadcrumbs"/);
      assert.doesNotMatch(html, /\{\{\{?[a-zA-Z0-9_.-]+/);
```

Add page-specific assertions:

```js
const architecture = await readFile(new URL("../../public/architecture/index.html", import.meta.url), "utf8");
const api = await readFile(new URL("../../public/api-compatibility/index.html", import.meta.url), "utf8");
const faq = await readFile(new URL("../../public/faq/index.html", import.meta.url), "utf8");
const comparison = await readFile(new URL("../../public/compare/omlx/index.html", import.meta.url), "utf8");
assert.match(architecture, /Swift-native in-process inference/);
assert.match(api, /curl http:\/\/localhost:8000\/v1\/chat\/completions/);
assert.equal((faq.match(/<details>/g) ?? []).length, 8);
assert.match(comparison, /official repository snapshot 2026-07-10/);
```

Run the test and expect FAIL because article outputs do not exist.

- [ ] **Step 2: Add article rendering to the builder**

In `scripts/build-public-site.mjs`, import the registries and renderer:

```js
import { facts } from "../site/content/facts.mjs";
import { competitors } from "../site/content/competitors.mjs";
import { faqs } from "../site/content/faqs.mjs";
import { releases } from "../site/content/releases.mjs";
import { renderBlocks } from "../site/lib/blocks.mjs";
import { validateDatedRegistry, validateFacts } from "../site/lib/content.mjs";
```

Build lookup maps once:

```js
const factsById = Object.fromEntries(facts.map((item) => [item.id, item]));
const competitorsById = Object.fromEntries(competitors.map((item) => [item.id, item]));
const faqsById = Object.fromEntries(faqs.map((item) => [item.id, item]));
const releasesById = Object.fromEntries(releases.map((item) => [item.id, item]));
```

At the start of `buildSite()`, before the asset copy or any write, run:

```js
validateFacts(facts, { today: project.lastVerified, maxAgeDays: 45 });
validateDatedRegistry(competitors, { label: "competitor", today: project.lastVerified, maxAgeDays: 45 });
validateDatedRegistry(releases, { label: "release", today: project.lastVerified, maxAgeDays: 45 });
for (const item of faqs) {
  if (!item.en?.question || !item.en?.answer || !item.zh?.question || !item.zh?.answer) {
    throw new Error(`incomplete FAQ: ${item.id}`);
  }
}
```

Import `escapeHTML` with `renderTokens`, then add these exact helpers after the lookup maps:

```js
const macmlxComparison = {
  en: {
    platform: "Apple Silicon macOS 14 or later.",
    runtime: "Swift-native in-process inference through Apple MLX.",
    models: "MLX language and vision-model repositories supported by MacMLXCore.",
    interfaces: "Native SwiftUI app, macmlx CLI, compatible HTTP APIs, and MCP surfaces.",
    bestFit: "Mac users prioritizing one inspectable Swift core across desktop, terminal, and API workflows.",
  },
  zh: {
    platform: "Apple 芯片 macOS 14 或更高版本。",
    runtime: "通过 Apple MLX 运行 Swift 原生进程内推理。",
    models: "MacMLXCore 支持的 MLX 语言与视觉模型仓库。",
    interfaces: "原生 SwiftUI 应用、macmlx CLI、兼容 HTTP API 与 MCP 接口。",
    bestFit: "重视桌面、终端和 API 共用一套可检查 Swift 核心的 Mac 用户。",
  },
};

const relatedIds = {
  architecture: ["api-compatibility", "models"],
  "api-compatibility": ["architecture", "faq"],
  faq: ["api-compatibility", "models"],
  models: ["choosing-a-model", "vision-language-models"],
  "choosing-a-model": ["models", "architecture"],
  "vision-language-models": ["models", "release-v0-5-3"],
  compare: ["compare-ollama", "compare-lm-studio", "compare-omlx"],
  "compare-ollama": ["compare", "architecture"],
  "compare-lm-studio": ["compare", "models"],
  "compare-omlx": ["compare", "architecture"],
  releases: ["release-v0-5-3", "architecture"],
  "release-v0-5-3": ["releases", "api-compatibility"],
};

function ui(locale) {
  return locale === "en"
    ? {
        home: "Home", architecture: "Architecture", api: "API", models: "Models", compare: "Compare", faq: "FAQ",
        download: "Download", skip: "Skip to content", verified: "Last verified:", sources: "Official sources",
        facts: "Verified facts", example: "Example", questions: "Questions and answers", related: "Related pages",
        relatedAria: "Related macMLX pages", released: "What shipped", next: "Development and planned boundaries",
      }
    : {
        home: "首页", architecture: "架构", api: "API", models: "模型", compare: "对比", faq: "常见问题",
        download: "下载", skip: "跳到正文", verified: "最后核验：", sources: "官方来源",
        facts: "已核验事实", example: "示例", questions: "问题与解答", related: "相关页面",
        relatedAria: "macMLX 相关页面", released: "已交付内容", next: "开发中与规划边界",
      };
}

function siteHeader(locale, counterpart) {
  const labels = ui(locale);
  const prefix = locale === "en" ? "" : "/zh";
  const otherLanguage = locale === "en" ? "中文" : "EN";
  const otherHreflang = locale === "en" ? "zh-Hans" : "en";
  return `<header class="site-header"><div class="nav-shell"><a class="wordmark" href="${prefix || "/"}"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span><span>macMLX</span></a><nav class="nav-links" aria-label="${locale === "en" ? "Primary navigation" : "主导航"}"><a href="${prefix}/architecture/">${labels.architecture}</a><a href="${prefix}/api-compatibility/">${labels.api}</a><a href="${prefix}/models/">${labels.models}</a><a href="${prefix}/compare/">${labels.compare}</a><a href="${prefix}/faq/">${labels.faq}</a></nav><div class="nav-actions"><button class="utility-button" id="theme-toggle" type="button" aria-label="${locale === "en" ? "Switch color theme" : "切换颜色主题"}"><svg class="sun-icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.5"></circle><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path></svg><svg class="moon-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 15.2A8.6 8.6 0 0 1 8.8 4a8.6 8.6 0 1 0 11.2 11.2Z"></path></svg></button><a class="language-link" href="${counterpart}" hreflang="${otherHreflang}">${otherLanguage}</a><a class="nav-download" href="${project.downloadURL}">${labels.download}</a></div></div></header>`;
}

function siteFooter(locale) {
  const labels = ui(locale);
  const prefix = locale === "en" ? "" : "/zh";
  return `<footer class="site-footer"><div class="page-shell footer-layout"><div><a class="wordmark footer-wordmark" href="${prefix || "/"}"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span><span>macMLX</span></a><p>${locale === "en" ? "Native Swift inference for Apple Silicon." : "面向 Apple 芯片的原生 Swift 推理。"}</p></div><div class="footer-links"><a href="${prefix}/architecture/">${labels.architecture}</a><a href="${prefix}/api-compatibility/">${labels.api}</a><a href="${prefix}/models/">${labels.models}</a><a href="${prefix}/compare/">${labels.compare}</a><a href="${prefix}/releases/">${locale === "en" ? "Releases" : "版本"}</a><a href="${project.repository}">${locale === "en" ? "Source" : "源码"}</a></div></div><div class="page-shell footer-bottom"><span>© 2026 macMLX contributors</span><span>${locale === "en" ? "Local by design. Open by choice." : "为本地而生，因开放而自由。"}</span></div></footer>`;
}

function breadcrumbHTML(page, locale) {
  const labels = ui(locale);
  const prefix = locale === "en" ? "" : "/zh";
  const pathSegments = page.paths[locale].split("/").filter(Boolean);
  const contentSegments = locale === "en" ? pathSegments : pathSegments.slice(1);
  const parentLabels = { models: labels.models, compare: labels.compare, releases: locale === "en" ? "Releases" : "版本" };
  const links = [`<a href="${prefix || "/"}">${labels.home}</a>`];
  if (contentSegments.length > 1) {
    links.push(`<span aria-hidden="true">/</span><a href="${prefix}/${contentSegments[0]}/">${parentLabels[contentSegments[0]]}</a>`);
  }
  links.push(`<span aria-hidden="true">/</span><span aria-current="page">${escapeHTML(page.title[locale])}</span>`);
  return `<nav class="breadcrumbs" aria-label="${locale === "en" ? "Breadcrumb" : "面包屑"}">${links.join("")}</nav>`;
}

function sourceItems(page) {
  const urls = new Set();
  for (const id of page.factIds || []) factsById[id].sourceUrls.forEach((url) => urls.add(url));
  if (page.competitorId) competitorsById[page.competitorId].officialSources.forEach((url) => urls.add(url));
  for (const id of page.competitorIds || []) competitorsById[id].officialSources.forEach((url) => urls.add(url));
  for (const id of page.faqIds || []) {
    for (const factId of faqsById[id].factIds) factsById[factId]?.sourceUrls.forEach((url) => urls.add(url));
  }
  if (page.releaseId) releasesById[page.releaseId].officialSources.forEach((url) => urls.add(url));
  for (const id of page.releaseIds || []) releasesById[id].officialSources.forEach((url) => urls.add(url));
  return [...urls].map((url, index) => ({ label: `Source ${index + 1} · ${new URL(url).hostname}`, url }));
}

function relatedBlock(page, locale) {
  const labels = ui(locale);
  return {
    type: "related",
    heading: labels.related,
    ariaLabel: labels.relatedAria,
    links: relatedIds[page.id].map((id) => {
      const route = pagesById[id];
      return { href: route.paths[locale], title: route.title[locale], description: route.metadata[locale].description };
    }),
  };
}

function blocksForPage(page, locale) {
  const labels = ui(locale);
  const copyLocale = locale === "en" ? "en" : "zh";
  const blocks = [];
  if (page.factIds?.length) blocks.push({ type: "facts", heading: labels.facts, factIds: page.factIds });
  if (page.table) blocks.push({ type: "table", ...page.table[locale] });
  if (page.code) blocks.push({ type: "code", heading: labels.example, intro: locale === "en" ? "Call the local compatible server." : "调用本地兼容服务。", code: page.code });
  if (page.faqIds?.length) blocks.push({ type: "faq", heading: labels.questions, faqIds: page.faqIds });
  const comparisonIds = page.competitorId ? [page.competitorId] : (page.competitorIds || []);
  for (const competitorId of comparisonIds) {
    const macmlx = macmlxComparison[copyLocale];
    blocks.push({
      type: "comparison",
      heading: page.competitorId ? page.title[locale] : `macMLX ${locale === "en" ? "and" : "与"} ${competitorsById[competitorId].name}`,
      competitorId,
      verifiedLabel: labels.verified,
      rows: [
        { key: "platform", label: locale === "en" ? "Platform" : "平台", macmlx: macmlx.platform },
        { key: "runtime", label: locale === "en" ? "Core runtime" : "核心运行时", macmlx: macmlx.runtime },
        { key: "models", label: locale === "en" ? "Model workflow" : "模型工作流", macmlx: macmlx.models },
        { key: "interfaces", label: locale === "en" ? "Interfaces" : "接口", macmlx: macmlx.interfaces },
        { key: "bestFit", label: locale === "en" ? "Best fit" : "适用人群", macmlx: macmlx.bestFit },
      ],
    });
  }
  if (page.releaseId) {
    const release = releasesById[page.releaseId];
    blocks.push({ type: "facts", heading: labels.released, factIds: release.factIds });
    blocks.push({ type: "paragraph", text: release[copyLocale].limitation });
    blocks.push({ type: "facts", heading: labels.next, factIds: release.developmentFactIds });
  }
  if (page.releaseIds?.length) {
    blocks.push({ type: "paragraph", text: locale === "en" ? "Open the version page for verified shipped facts, limitations, and official release links." : "打开具体版本页面，查看已核验的交付事实、限制与官方发布链接。" });
  }
  const sources = sourceItems(page);
  if (sources.length) blocks.push({ type: "sources", heading: labels.sources, sources });
  blocks.push(relatedBlock(page, locale));
  return blocks;
}

function temporaryArticleHead(page, locale) {
  const metadata = page.metadata[locale];
  const canonical = canonicalURL(project.origin, page.paths[locale]);
  const english = canonicalURL(project.origin, page.paths.en);
  const chinese = canonicalURL(project.origin, page.paths["zh-Hans"]);
  return `<title>${escapeHTML(metadata.title)}</title><meta name="description" content="${escapeHTML(metadata.description)}"><link rel="canonical" href="${canonical}"><link rel="alternate" hreflang="en" href="${english}"><link rel="alternate" hreflang="zh-Hans" href="${chinese}"><link rel="alternate" hreflang="x-default" href="${english}">`;
}

function renderArticle(template, page, locale) {
  const labels = ui(locale);
  const content = renderBlocks(blocksForPage(page, locale), {
    copyLocale: locale === "en" ? "en" : "zh",
    factsById,
    faqsById,
    competitorsById,
  });
  return renderTokens(template, {
    htmlLang: project.htmlLanguages[locale],
    bodyLanguage: locale === "en" ? "en" : "zh",
    skipLabel: labels.skip,
    head: temporaryArticleHead(page, locale),
    siteHeader: siteHeader(locale, counterpartPath(page, locale)),
    breadcrumbs: breadcrumbHTML(page, locale),
    eyebrow: page.eyebrow[locale],
    title: page.title[locale],
    directAnswer: page.directAnswer[locale],
    verifiedLabel: labels.verified,
    lastVerified: page.lastVerified,
    content,
    siteFooter: siteFooter(locale),
  }, new Set(["head", "siteHeader", "breadcrumbs", "content", "siteFooter"]));
}
```

Also define:

```js
const pagesById = Object.fromEntries(routes.filter((item) => item.kind === "article").map((item) => [item.id, item]));
```

Read `site/templates/article.html` once. After home generation, execute:

```js
const articleTemplate = await readFile(resolve(repositoryRoot, "site/templates/article.html"), "utf8");
for (const page of routes.filter((item) => item.kind === "article")) {
  for (const locale of project.locales) {
    const output = renderArticle(articleTemplate, page, locale);
    const outputPath = resolve(repositoryRoot, "public", outputFileForPath(page.paths[locale]));
    await mkdir(dirname(outputPath), { recursive: true });
    await writeFile(outputPath, output.endsWith("\n") ? output : `${output}\n`, "utf8");
  }
}
```

- [ ] **Step 3: Add the shared editorial CSS**

Append focused rules to `site/assets/css/main.css` for:

```css
.article-main { padding: 140px 0 100px; }
.article-shell { max-width: 980px; }
.breadcrumbs { display: flex; flex-wrap: wrap; gap: 8px; color: var(--ink-faint); font: 10px/1.4 var(--mono); }
.breadcrumbs a { border-bottom: 1px solid var(--line-strong); }
.article-hero { padding: 72px 0 64px; border-bottom: 1px solid var(--line); }
.article-eyebrow, .verified-note { color: var(--ink-faint); font: 10px/1.5 var(--mono); letter-spacing: .08em; text-transform: uppercase; }
.article-hero h1 { max-width: 900px; margin: 18px 0 28px; font-size: clamp(52px, 7vw, 92px); line-height: .94; letter-spacing: -.065em; }
.article-answer { max-width: 760px; color: var(--ink-soft); font-size: 20px; line-height: 1.6; }
.article-body { max-width: 820px; margin: 0 auto; }
.content-section { padding: 64px 0; border-bottom: 1px solid var(--line); }
.content-section h2 { margin: 0 0 26px; font-size: clamp(30px, 4vw, 48px); letter-spacing: -.045em; }
.fact-cards { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.fact-card { padding: 24px; border: 1px solid var(--line); border-radius: 18px; background: var(--paper-deep); }
.fact-card__meta { display: flex; justify-content: space-between; color: var(--ink-faint); font: 9px/1.4 var(--mono); text-transform: uppercase; }
.fact-card h3 { margin: 28px 0 12px; font-size: 24px; letter-spacing: -.035em; }
.fact-card p, .article-paragraph { color: var(--ink-soft); font-size: 16px; line-height: 1.7; }
.article-code { padding: 24px; color: var(--demo-text); background: var(--demo); border-radius: 16px; overflow: auto; }
.comparison-wrap { overflow: visible; }
.comparison-table, .article-table { width: 100%; border-collapse: collapse; }
.comparison-table caption, .article-table caption { margin-bottom: 18px; text-align: left; color: var(--ink-soft); }
.comparison-table th, .comparison-table td, .article-table th, .article-table td { padding: 18px; border: 1px solid var(--line); text-align: left; vertical-align: top; line-height: 1.55; }
.faq-list details { border-top: 1px solid var(--line); padding: 20px 0; }
.faq-list summary { cursor: pointer; font-weight: 700; }
.faq-list p { color: var(--ink-soft); line-height: 1.7; }
.sources a { color: var(--blue); border-bottom: 1px solid currentColor; }
.related-pages { padding: 64px 0; }
.related-pages > div { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
.related-pages a { padding: 22px; border: 1px solid var(--line); border-radius: 16px; }
.related-pages span { display: block; margin-top: 8px; color: var(--ink-soft); line-height: 1.5; }
```

At `max-width: 720px`, use one-column facts and related links. Convert comparison rows into grid cards without horizontal scrolling:

```css
.comparison-table, .comparison-table tbody, .comparison-table tr, .comparison-table th, .comparison-table td,
.article-table, .article-table tbody, .article-table tr, .article-table th, .article-table td { display: block; }
.comparison-table thead, .article-table thead { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
.comparison-table tr, .article-table tr { margin-bottom: 14px; border: 1px solid var(--line); border-radius: 16px; overflow: hidden; }
.comparison-table th, .comparison-table td, .article-table th, .article-table td { border: 0; border-top: 1px solid var(--line); }
.comparison-table th:first-child, .article-table th:first-child { border-top: 0; background: var(--paper-deep); }
```

- [ ] **Step 4: Build and run all content checks**

Run:

```bash
node scripts/build-public-site.mjs
node --test site/tests/localize.test.mjs site/tests/routes.test.mjs site/tests/content.test.mjs site/tests/blocks.test.mjs site/tests/build-home.test.mjs site/tests/build-content.test.mjs
node scripts/test-public-site.mjs
git diff --check
```

Expected: all tests pass; 26 canonical locale routes exist; no horizontal-scroll CSS exception is introduced.

- [ ] **Step 5: Commit the generated-hub implementation**

```bash
git add site scripts/build-public-site.mjs scripts/test-public-site.mjs
git commit -m "feat(site): publish the bilingual technical content hub" \
  -m "Constraint: Public output remains static, dependency-free, and separately deployed" \
  -m "Confidence: medium" \
  -m "Scope-risk: broad" \
  -m "Directive: Add new pages through registries and route manifests, never one-off HTML" \
  -m "Tested: full Node site suite and static build"
```

### Task 6: Browser-Verify the Knowledge and Comparison Experience

**Files:**
- Create: `output/qa/seo-geo-content/architecture-en-light.png`
- Create: `output/qa/seo-geo-content/architecture-zh-dark.png`
- Create: `output/qa/seo-geo-content/compare-omlx-en-light.png`
- Create: `output/qa/seo-geo-content/compare-omlx-zh-mobile.png`
- Create: `.omx/state/seo-geo-content/ralph-progress.json`

- [ ] **Step 1: Serve the generated output and test representative routes**

Verify:

```text
1440 x 900 /architecture/ English light and dark
1440 x 900 /zh/architecture/ Chinese light and dark
1440 x 900 /compare/omlx/ English light
390 x 844 /zh/compare/omlx/ Chinese light and dark
390 x 844 /zh/faq/ Chinese with all eight answers readable
```

At every sample confirm:

```text
exact language counterpart link
visible last-verified date
working breadcrumbs and related links
zero horizontal overflow
zero console errors
zero failed formal resources
comparison source links reach official domains
```

- [ ] **Step 2: Capture and run visual-verdict**

Capture the four named screenshots. Compare the article pages to the approved landing-page visual system, using category hint `premium bilingual technical-product editorial hub`. Require a score of at least 90 and persist the standard verdict fields plus screenshot hashes in `.omx/state/seo-geo-content/ralph-progress.json`.

- [ ] **Step 3: Test no-JavaScript and reduced-motion behavior**

Disable or force the relevant behavior using the browser QA method. Confirm all body copy, tables/cards, FAQ answers, breadcrumbs, language links, and related links remain available. Record the method and restore source hashes immediately.

- [ ] **Step 4: Run the plan completion gate**

Run:

```bash
node scripts/build-public-site.mjs
node --test site/tests/*.test.mjs
node scripts/test-public-site.mjs
node --check public/assets/js/main.js
git diff --check
```

Expected: all checks pass and visual-verdict is at least 90.

## Content Hub Completion Gate

- All 13 route IDs produce 26 canonical HTML pages.
- Every page has a visible direct answer and verification date.
- All reusable claims come from registries.
- Competitor pages name official sources and dated snapshots.
- FAQ, code, facts, comparisons, sources, and related links render semantically.
- Desktop/mobile and light/dark visual QA passes without horizontal overflow.
- No-JavaScript and reduced-motion flows preserve all knowledge content.
