import Foundation
import MLXLLM
import MLXLMCommon

/// Registration hook for **macMLX-owned model architectures** — pure-Swift
/// implementations of model families that upstream `mlx-swift-lm` does not
/// (yet) ship.
///
/// ## Why this exists
///
/// `mlx-swift-lm`'s `LLMTypeRegistry.shared` is a `public actor` with a
/// `public registerModelType(_:creator:)` method, and `LLMModelFactory.shared`
/// holds a reference to that same shared registry. So we can teach the
/// stock factory about a new `model_type` **from outside the package** — no
/// fork, no patch, no merge pain. Apple ships this exact pattern themselves
/// (`Gemma4AssistantRegistration` in MLXVLM registers into MLXLMCommon's
/// registry across a module boundary).
///
/// When we implement, say, DeepSeek V4 as a pure-Swift `LLMModel` in
/// `MacMLXCore/Models/`, we register it here. `MLXSwiftEngine.load(_:)`
/// then resolves `config.json`'s `model_type: deepseek_v4` to our type
/// automatically — the engine's existing `LLMModelFactory.shared
/// .loadContainer(from:)` path is unchanged.
///
/// ## Lifecycle
///
/// `registerAll()` is idempotent and called once, lazily, before the first
/// model load (see `MLXSwiftEngine`). Re-registering a `model_type`
/// overwrites the prior creator with the same one, so double-calls are safe.
///
/// ## Upstream contribution
///
/// Architectures registered here are candidates for upstreaming to
/// `mlx-swift-lm`. When upstream lands one, delete our implementation +
/// its registration; the stock factory picks it up. This keeps the overlay
/// a *thin, shrinking* layer rather than a divergent fork.
public enum ModelOverlay {

    /// Register every macMLX-owned architecture into the shared factory
    /// registry. Idempotent: re-registering a `model_type` overwrites the
    /// prior creator with the same closure, so double-calls are safe.
    ///
    /// - Note: `async` because `ModelTypeRegistry` is an `actor`.
    public static func registerAll() async {
        // DeepSeek V3.2 — pure-Swift port (see `Models/DeepseekV32.swift`).
        // The creator closure is inlined because the package's `create`
        // helper is `private`. `JSONDecoder.json5()` matches the decoder the
        // stock factory uses for `config.json` (tolerant of comments/trailing
        // commas), mirroring `MTPDrafterModelFactory` in mlx-swift-lm.
        await LLMTypeRegistry.shared.registerModelType("deepseek_v32") { data in
            let config = try JSONDecoder.json5()
                .decode(DeepseekV32Configuration.self, from: data)
            return DeepseekV32Model(config)
        }

        // Mellum 2 (12B-A2.5B) — pure-Swift port (see `Models/Mellum2.swift`).
        // A Qwen3-lineage sparse-MoE decoder with alternating sliding/full
        // attention; upstream mlx-swift-lm has no `mellum` type.
        await LLMTypeRegistry.shared.registerModelType("mellum") { data in
            let config = try JSONDecoder.json5()
                .decode(Mellum2Configuration.self, from: data)
            return Mellum2Model(config)
        }

        // Seed-OSS (ByteDance Seed-OSS-36B) — pure-Swift port (see
        // `Models/SeedOss.swift`). A dense Llama-family decoder whose only
        // architecture-specific twists are three bias switches (attention_bias,
        // the SEPARATE attention_out_bias, and mlp_bias); upstream mlx-swift-lm
        // has no `seed_oss` type.
        await LLMTypeRegistry.shared.registerModelType("seed_oss") { data in
            let config = try JSONDecoder.json5()
                .decode(SeedOssConfiguration.self, from: data)
            return SeedOssModel(config)
        }

        // --- Theoretical tier -------------------------------------------------
        // Near-zero-engineering registrations: each maps a new `model_type`
        // onto an existing, parity-tested Swift architecture. They are verified
        // at the code layer via fixture parity (config parsing / sanitize /
        // difference components) but never smoke-tested on real weights — the
        // smallest quant of each exceeds the size budget. Model-specific issues
        // are handled issue-driven. See the per-model files for the mapping and
        // the exact difference surface each fixture covers.

        // Solar-Open-100B (`solar_open`) → upstream `GLM4MoEModel`, unchanged.
        // `solar_open.py` is a config re-skin of `glm4_moe`; the only work is
        // injecting the ModelArgs defaults Solar's config.json omits but
        // GLM4MoEConfiguration requires. See `Models/SolarOpen.swift`.
        await LLMTypeRegistry.shared.registerModelType("solar_open") { data in
            let patched = try SolarOpen.patchedConfigData(data)
            let config = try JSONDecoder.json5()
                .decode(GLM4MoEConfiguration.self, from: patched)
            return GLM4MoEModel(config)
        }

        // GLM-DSA (`glm_moe_dsa`, GLM-5.1) → our own `DeepseekV32Model`.
        // `glm_moe_dsa.py` is `class Model(DSV32Model)`; `GlmMoeDsaConfiguration`
        // applies the GLM-DSA ModelArgs defaults (interleaved indexer RoPE;
        // rope derived from `rope_parameters`) and rejects GLM-5.2 IndexShare
        // configs loudly. See `Models/GlmMoeDsa.swift`.
        await LLMTypeRegistry.shared.registerModelType("glm_moe_dsa") { data in
            let config = try JSONDecoder.json5()
                .decode(GlmMoeDsaConfiguration.self, from: data)
            return DeepseekV32Model(config.base)
        }

        // Kimi K2.5 (`kimi_k25`) → DeepSeek V3 text core. BLOCKED: upstream
        // `DeepseekV3Model.init` is `internal`, so the overlay cannot construct
        // the text tower to wrap. The config half is complete, so decode it
        // (proving that path) and then fail loudly rather than silently. See
        // `Models/KimiK25.swift`.
        await LLMTypeRegistry.shared.registerModelType("kimi_k25") { data in
            _ = try JSONDecoder.json5()
                .decode(KimiK25Configuration.self, from: data)
            throw ModelOverlayError.kimiK25RequiresPublicDeepseekV3Init
        }
    }
}
