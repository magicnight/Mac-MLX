// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon

/// Converts a model's freshly created stock KV caches into ragged
/// ``BatchKVCache`` instances for a left-padded cohort — the Swift analogue of
/// mlx-lm's `to_batch_cache` (`generate.py`), restricted to the v1 dense-only
/// scope.
///
/// ## Dense-only gate (mirrors A1's `batchPositioned`)
/// Python's `to_batch_cache` also maps `RotatingKVCache → BatchRotatingKVCache`.
/// The rotating sibling is deferred (see ``BatchKVCache``'s "Deferred to a
/// follow-up wave" note), so this converter REFUSES — returns `nil` for the
/// whole cohort — the moment it sees a
/// cache that is not a plain `KVCacheSimple` (sliding-window `RotatingKVCache`,
/// `ChunkedKVCache`, `QuantizedKVCache`, hybrid `CacheList`, SSM `MambaCache`).
/// Refusing wholesale, rather than converting the dense layers and leaving the
/// rest stock, keeps the cohort's caches a single coherent kind — a mixed set
/// would desynchronise per-row offsets and masks. Callers route a refused model
/// through the non-batched path (exactly as A1's gate does).
///
/// This is standalone A2c infrastructure: it is the seam a ragged scheduler
/// calls after `model.newCache(parameters:)`, not yet wired into
/// ``BatchDecodeRunner`` (whose equal-length A2a path is unchanged).
public enum BatchCacheConverter {
    /// Build one ``BatchKVCache`` per layer for a cohort left-padded by
    /// `leftPadding`, or `nil` if any input cache is not a plain dense
    /// `KVCacheSimple`.
    ///
    /// - Parameters:
    ///   - caches: the per-layer caches from `model.newCache(parameters:)` (must
    ///     be freshly created / empty; `BatchKVCache` starts from an empty buffer).
    ///   - leftPadding: the per-row pad amounts for the cohort (shared across all
    ///     layers), e.g. from ``BatchPrefillAssembly/leftPad(prompts:padToken:)``.
    /// - Returns: a fresh `BatchKVCache` per layer, or `nil` to signal the model
    ///   cannot be ragged-batched in v1.
    public static func makeBatchCaches(
        from caches: [KVCache], leftPadding: [Int]
    ) -> [KVCache]? {
        // Validate every layer FIRST, then build. Refusing before constructing
        // any `BatchKVCache` keeps the refusal path allocation-free (and MLX-free
        // — it only performs type checks), so a mixed cohort never leaves a
        // half-built result and never touches the Metal backend just to say no.
        // Exact-type check on purpose: `is KVCacheSimple` would admit subclasses
        // (`ChunkedKVCache` today, anything a future dependency bump adds), and a
        // subclass's extra semantics would be silently dropped by the dense
        // batch conversion. Only a plain `KVCacheSimple` is convertible.
        for cache in caches where type(of: cache) != KVCacheSimple.self {
            return nil
        }
        return caches.map { _ in BatchKVCache(leftPadding: leftPadding) }
    }
}
