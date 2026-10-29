// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon

/// Model-architecture coverage gate for the batched-decode path — the second
/// half of the A2 coverage probe (the first being the dense-cache gate in
/// ``BatchDecodeRunner`` / ``BatchCacheConverter``, which prove the cache SHAPE
/// is safe).
///
/// ## Why a type gate is necessary on top of the cache gate
/// Ragged batching (A2b) positions rows through the per-row `.batch` offset,
/// which only helps models whose forward reads the decode offset via
/// `cache.ropeOffset`. Several `MLXVLM` text towers instead read the SCALAR
/// `cache.offset` directly, so a per-row offset is a SILENT no-op for them: no
/// crash, but it is ignored and ragged decode corrupts. The cache gate is
/// therefore *necessary but not sufficient*; this predicate adds the missing
/// architecture check, restricting batching to the dense text models whose
/// `ropeOffset` wiring is verified in the vendored mlx-swift-lm.
///
/// ## The verified-covered set
/// The `MLXLLM` dense-attention text models confirmed to read `ropeOffset`
/// AND to take every layer's attention mask from the cache (`cache.makeMask`,
/// directly or through the single-cache `createAttentionMask(h:cache:)`):
/// Gemma / Gemma3Text / Gemma4Text / Qwen2 / Qwen3 / Llama. The second
/// condition matters because ``BatchKVCache/makeMask(n:windowSize:returnArray:)``
/// is what hides each row's left-pad columns; a model that builds its mask
/// without asking the cache attends to them. The match is on the concrete model
/// type's bare name (`String(describing:)`): an unlisted architecture is refused
/// wholesale, and the caller routes it through the non-batched path.
///
/// `BatchRaggedDecodeParityTests` checks each listed architecture against
/// standalone decode on a ragged cohort; add an architecture there when adding
/// it here.
///
/// A string match (rather than `is GemmaModel`) keeps this file free of a
/// concrete `import MLXLLM` dependency and, more importantly, fails CLOSED for
/// any model this list has not vetted — a new architecture is refused until a
/// human adds it here, which is the safe default for a silent-no-op hazard.
public enum BatchModelAllowlist {
    /// Concrete model type names verified to read `cache.ropeOffset` (and thus
    /// honour A1's per-row `.batch` offset) and to build every mask through
    /// `cache.makeMask` in the vendored mlx-swift-lm.
    ///
    /// NOT `Gemma2Model`: it reads `ropeOffset`, but `Gemma2ModelInner` builds
    /// its mask with the `[KVCache]?` overload of `createAttentionMask`, which
    /// never calls `makeMask` (nil at one token, a plain causal mask from the
    /// first cache's offset otherwise), so in a ragged batch the left-padded
    /// rows attend to their zero-filled pad columns. Wrong logits, no error.
    /// Re-add it only once that model takes its mask from the cache and
    /// `BatchRaggedDecodeParityTests.testGemma2RaggedDecodeAttendsToLeftPadding`
    /// fails as a result.
    private static let allowlisted: Set<String> = [
        "GemmaModel",
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
