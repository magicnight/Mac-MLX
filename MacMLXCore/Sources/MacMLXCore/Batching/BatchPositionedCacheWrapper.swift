// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// `MLXFast.ScaledDotProductAttentionMaskMode` is reachable through `MLX`/`MLXNN`
// (matches upstream `KVCache.swift`, which references it with only those
// imports); `MLXFast` is not a separately importable product here.

/// Batch-positioned KV-cache wrapper — the correctness fix for mlx-swift's
/// batched single-token decode (continuous batching, Track A step 1).
///
/// ## What it does
/// Wraps a stock ``KVCache`` (`KVCacheSimple`/`StandardKVCache` for global
/// layers, `RotatingKVCache` for sliding layers) and forwards EVERY `KVCache`
/// requirement verbatim to the inner cache, changing exactly ONE observable
/// behavior: it conforms to ``BatchPositionedKVCache``, so `ropeOffset` returns
/// `.batch([offset × B])` (a per-row array offset) instead of the default
/// `.scalar(offset)`.
///
/// ## Why it exists (the bug)
/// mlx-swift 0.31.6's `mx.fast.rope` corrupts batched single-token decode: for a
/// query shaped `[B, H, 1, D]` with `B > 1` and a SCALAR offset, lanes `1..<B`
/// read uninitialized memory (NaN / garbage), so identical prompt rows diverge
/// instead of emitting identical tokens. This is upstream mlx-core bug
/// ml-explore/mlx#3494 / #3496, fixed in mlx-core 0.32.0 (PR #3498). mlx-swift
/// 0.31.6 still vendors mlx-core v0.31.1, so the fix is not yet available from
/// the Swift side; we filed the mlx-swift vendor-bump request as
/// ml-explore/mlx-swift#441.
///
/// ## How the fix works — and its real scope
/// Routing RoPE through a `.batch` (array) offset takes the per-row array-offset
/// kernel path (`applyRotaryPosition`'s `.batch` case → `rope(x, offset:
/// MLXArray)`), which is correct even at `B > 1`. This helps ONLY models whose
/// forward reads the decode offset via `cache?.ropeOffset`. Verified COVERED —
/// the `MLXLLM` dense-attention text models `Gemma`, `Gemma2`, `Gemma3Text`,
/// `Gemma4Text`, `Qwen2`, `Qwen3`, and `Llama` all do
/// `let offset = cache?.ropeOffset; … applyRotaryPosition(rope, to:, offset:)`.
///
/// Verified NOT covered — several `MLXVLM` text towers read the SCALAR
/// `cache.offset` directly, bypassing `ropeOffset` entirely: `Paligemma`,
/// `Gemma3` (the VLM attention in `MLXVLM/Models/Gemma3.swift`, distinct from
/// the covered `MLXLLM/Models/Gemma3Text.swift`), `LFM2VL`, `Pixtral`,
/// `Mistral3`, and `Gemma4` (likewise the VLM file, distinct from the covered
/// `Gemma4Text`). Wrapping their cache is not harmful (no crash — see
/// ``batchPositioned(_:batch:)`` for the crash-causing cases) but it is a
/// silent NO-OP: the batched-decode bug remains uncorrected because the model
/// never reads the `.batch` offset this wrapper installs.
///
/// This wrapper has NOT been audited against every model in the vendored
/// mlx-swift-lm — only the list above. Callers must confirm a target model
/// reads `ropeOffset` (not `offset`) before relying on this fix for
/// correctness. See ``batchPositioned(_:batch:)``, which is agnostic across
/// cache TYPES it can prove safe, not across every model's RoPE wiring.
///
/// Why an ARRAY of EQUAL offsets works: aligned identical prompts sit at the
/// same position, so `batchOffset = [offset, offset, …]`. A primitive probe
/// proved that an array of equal values already takes the correct (non-buggy)
/// kernel path (cross-row 0.0, row0 == the `B == 1` reference), and that a
/// per-row DIFFERENT offset array is also numerically correct — exactly what
/// real continuous batching needs once sequences desynchronize.
///
/// ## Composition, not subclassing
/// `KVCacheSimple`/`RotatingKVCache` are declared `public` (not `open`), so they
/// cannot be subclassed. This wrapper forwards to a held instance instead. That
/// makes it a safe drop-in ONLY when the wrapped inner cache is one of those two
/// plain dense types. Some models reach their cache through a concrete-type cast
/// this wrapper cannot satisfy (hybrid models do `cache as? CacheList`, then
/// `cacheList[0] as? MambaCache`) or a capability probe whose fallback path
/// crashes for the specific cache being probed (`cache as? QuantizedKVCacheProtocol`
/// — harmless to fail for a plain dense cache, but wrapping an actual
/// `QuantizedKVCache` makes that probe fail and fall through to a `fatalError`
/// in its non-quantized `update`). Use ``batchPositioned(_:batch:)`` rather than
/// this initializer directly — it enforces the dense-type restriction and
/// refuses (returns `nil`) otherwise; this initializer itself does not validate
/// its input.
///
/// ## KV sharing
/// Handled for free: a model's `newCache` only creates caches for the KV-owning
/// layers; shared layers (e.g. Gemma-4 `num_kv_shared_layers`) receive a `nil`
/// cache plus the owning layer's `positionOffset`, which the forward propagates
/// verbatim. Wrapping the owning caches therefore makes every layer — owning and
/// shared — take the `.batch` RoPE path.
///
/// ## Deletable
/// This wrapper is a temporary shim over an upstream defect. Once mlx-swift
/// vendors mlx-core ≥ 0.32.0 (ml-explore/mlx-swift#441), the SCALAR path is
/// correct and this type — plus ``batchPositioned(_:batch:)`` — can be deleted.
/// The regression test `BatchPositionedCacheWrapperTests` is the tripwire: its
/// "scalar path is broken" assertion starts failing when the upstream fix lands,
/// signaling that the shim is no longer needed.
///
/// - Note: Standalone infrastructure. Not yet wired into any generation path;
///   the continuous-batching scheduler (Track A step 2) is the first consumer.
public final class BatchPositionedCacheWrapper: BatchPositionedKVCache {
    /// The wrapped stock cache. `var` because `KVCache` is not class-constrained,
    /// so the `{ get set }` protocol properties (`state`, `metaState`) require a
    /// mutable base to satisfy the setter witness; the held value is always a
    /// class instance in practice.
    private var inner: KVCache

    /// Number of sequences (rows) in the assembled batch, `B`.
    private let batch: Int

    /// Wrap `inner`, exposing a per-row `.batch` RoPE offset for a `batch`-row
    /// batched decode.
    public init(wrapping inner: KVCache, batch: Int) {
        self.inner = inner
        self.batch = batch
    }

    // MARK: - BatchPositionedKVCache

    /// Per-sequence RoPE offsets, shape `[B]`. Aligned rows ⇒ the current scalar
    /// offset replicated `batch` times.
    ///
    /// Read at the CURRENT (pre-`update`) offset because models snapshot
    /// `cache?.ropeOffset` before calling `cache.update(...)`. Snapshot safety
    /// comes from this computed property itself: `Int32(truncatingIfNeeded:)`
    /// eagerly reads `inner.offset` as a host `Int` and copies it into a
    /// brand-new `MLXArray`, so a later mutation of `inner.offset` cannot
    /// retroactively change an array already handed to a caller. (The
    /// ``BatchPositionedKVCache`` protocol extension additionally wraps this as
    /// `.batch(batchOffset + 0)` for `ropeOffset`, but that `+ 0` is just the
    /// protocol's uniform wrapping — it is not what makes this safe.)
    /// `truncatingIfNeeded` avoids a trapping conversion: `offset` is a host
    /// `Int` that in ordinary use will never approach `Int32.max` decoded
    /// tokens, but a non-trapping conversion is the right default for a value
    /// crossing the host/array boundary rather than asserting an invariant
    /// nothing here enforces.
    public var batchOffset: MLXArray {
        MLXArray(Array(repeating: Int32(truncatingIfNeeded: inner.offset), count: batch))
    }

    // MARK: - KVCache passthrough

    public var offset: Int { inner.offset }

    public var maxSize: Int? { inner.maxSize }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        inner.update(keys: keys, values: values)
    }

    public var state: [MLXArray] {
        get { inner.state }
        set { inner.state = newValue }
    }

    public var metaState: [String] {
        get { inner.metaState }
        set { inner.metaState = newValue }
    }

    public var isTrimmable: Bool { inner.isTrimmable }

    @discardableResult
    public func trim(_ n: Int) -> Int { inner.trim(n) }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        inner.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public func copy() -> any KVCache {
        BatchPositionedCacheWrapper(wrapping: inner.copy(), batch: batch)
    }

    public func prepare(lengths: [Int]?) { inner.prepare(lengths: lengths) }

    public func prepare(lengths: MLXArray?) { inner.prepare(lengths: lengths) }

    public func finalize() { inner.finalize() }

    // MARK: - Evaluatable

    public func innerState() -> [MLXArray] { inner.innerState() }
}

/// Wrap a model's freshly created KV caches so batched single-token decode takes
/// the correct per-row RoPE kernel path — or refuse if that cannot be done
/// safely.
///
/// This is the entry point for the ``BatchPositionedCacheWrapper`` fix. The
/// continuous-batching scheduler (Track A step 2) calls this immediately after
/// `model.newCache(parameters:)` when it assembles a `batch`-row batch:
///
/// ```swift
/// guard let caches = batchPositioned(model.newCache(parameters: params), batch: B) else {
///     // This model's caches cannot be safely batch-positioned — do not
///     // continuous-batch it (fall back to sequential / batch == 1 decode).
///     ...
/// }
/// ```
///
/// ## Safety across cache types, not coverage across models
/// This function is agnostic across cache TYPES: it wraps ONLY the two dense,
/// non-composite cache types the wrapper is proven safe for —
/// `KVCacheSimple`/`StandardKVCache` and `RotatingKVCache`. Any other cache type
/// makes the WHOLE call return `nil` rather than wrapping the safe ones and
/// silently leaving the rest unwrapped, because the failure modes for the rest
/// are not obviously loud:
///
///  - **Crash risk** — `CacheList` (hybrid models, e.g. `FalconH1`/`BaichuanM1`
///    via `CacheList(MambaCache(), attentionCache)`) implements `update` as
///    `fatalError(...)` and models reach its children via
///    `cache?[0] as? MambaCache` subscripting. Wrapping the `CacheList` itself
///    both breaks that subscript access (wrong concrete type) and crashes if
///    `update` is ever invoked on it. `QuantizedKVCache` similarly implements
///    `update` as `fatalError` (its real path is `updateQuantized`, reached by
///    models via `cache as? QuantizedKVCacheProtocol` — a probe this wrapper
///    deliberately does not satisfy). Wrapping a real `QuantizedKVCache` defeats
///    that probe and falls through to the crashing `update`.
///  - **Silent no-op risk** — even a successfully wrapped dense cache fixes
///    nothing for a model that reads the SCALAR `cache.offset` instead of
///    `cache.ropeOffset`. See the "How the fix works — and its real scope"
///    section on ``BatchPositionedCacheWrapper`` for the verified-covered vs.
///    verified-bypassing model lists. This case is model behavior, not cache
///    shape, so it CANNOT be detected here — callers must independently confirm
///    the target model reads `ropeOffset` before trusting a non-`nil` result.
///
/// - Parameters:
///   - caches: The per-layer KV caches from `model.newCache(parameters:)`.
///   - batch: The number of sequences (rows) `B` in the assembled batch.
/// - Returns: Every cache wrapped in a ``BatchPositionedCacheWrapper``, or
///   `nil` if any cache is not a `KVCacheSimple`/`RotatingKVCache` — meaning
///   this model's cache shape cannot be safely batch-positioned by this
///   function.
public func batchPositioned(_ caches: [KVCache], batch: Int) -> [KVCache]? {
    var wrapped: [KVCache] = []
    wrapped.reserveCapacity(caches.count)
    for cache in caches {
        guard cache is KVCacheSimple || cache is RotatingKVCache else {
            return nil
        }
        wrapped.append(BatchPositionedCacheWrapper(wrapping: cache, batch: batch))
    }
    return wrapped
}
