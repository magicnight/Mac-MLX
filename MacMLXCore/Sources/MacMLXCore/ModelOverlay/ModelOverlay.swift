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
    }
}
