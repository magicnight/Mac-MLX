// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon

/// Model-architecture coverage gate for the batched-decode path — the second
/// half of the A2 coverage probe (the first being ``batchPositioned(_:batch:)`` /
/// ``BatchCacheConverter``, which prove the cache SHAPE is safe).
///
/// ## Why a type gate is necessary on top of the cache gate
/// The A1 RoPE fix (per-row `.batch` offset) only helps models whose forward
/// reads the decode offset via `cache.ropeOffset`. Several `MLXVLM` text towers
/// instead read the SCALAR `cache.offset` directly (see
/// ``BatchPositionedCacheWrapper`` "How the fix works — and its real scope"),
/// so a batched cache is a SILENT no-op for them: no crash, but the per-row
/// offset is ignored and ragged decode corrupts. A non-nil cache gate is
/// therefore *necessary but not sufficient*; this predicate adds the missing
/// architecture check, restricting batching to the dense text models whose
/// `ropeOffset` wiring is verified in the vendored mlx-swift-lm.
///
/// ## The verified-covered set
/// The `MLXLLM` dense-attention text models confirmed to read `ropeOffset`:
/// Gemma / Gemma2 / Gemma3Text / Gemma4Text / Qwen2 / Qwen3 / Llama. The match
/// is on the concrete model type's bare name (`String(describing:)`), mirroring
/// how ``batchPositioned(_:batch:)`` is agnostic across cache types it can prove
/// safe: an unlisted architecture is refused wholesale, and the caller routes
/// it through the non-batched path.
///
/// A string match (rather than `is GemmaModel`) keeps this file free of a
/// concrete `import MLXLLM` dependency and, more importantly, fails CLOSED for
/// any model this list has not vetted — a new architecture is refused until a
/// human adds it here, which is the safe default for a silent-no-op hazard.
public enum BatchModelAllowlist {
    /// Concrete model type names verified to read `cache.ropeOffset` (and thus
    /// honour A1's per-row `.batch` offset) in the vendored mlx-swift-lm.
    private static let allowlisted: Set<String> = [
        "GemmaModel",
        "Gemma2Model",
        "Gemma3TextModel",
        "Gemma4TextModel",
        "Qwen2Model",
        "Qwen3Model",
        "LlamaModel",
    ]

    /// Whether `model`'s architecture is on the verified-`ropeOffset` allowlist,
    /// i.e. batched ragged decode is numerically safe for it. Refuses (returns
    /// `false`) for every unlisted model so the scheduler falls back rather than
    /// silently emitting corrupt output on a `cache.offset`-reading tower.
    public static func contains(_ model: any LanguageModel) -> Bool {
        allowlisted.contains(String(describing: type(of: model)))
    }
}
