// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// `MLXFast.ScaledDotProductAttentionMaskMode` is reachable through `MLX`/`MLXNN`
// (matches upstream `KVCache.swift`); `MLXFast` is not separately importable here.

/// Ragged (left-padded) batched KV cache — the pure-Swift port of mlx-lm's
/// `BatchKVCache` (`mlx_lm/models/cache.py:912`, main), the piece that lets a
/// cohort of DIFFERENT-length prompts share one `[B, H, S, D]` cache.
///
/// ## Why stock caches can't do this (A2b, the ragged upgrade over A2a)
/// `KVCacheSimple` holds ONE scalar `offset` for all `B` rows and its
/// `makeMask` ignores left-padding, so a cohort of unequal prompt lengths
/// left-padded to a common width `Lmax` attends its pad positions and corrupts
/// attention. `BatchKVCache` fixes both:
///  - **Per-row RoPE offset** (`batchOffset`, shape `[B]`) initialised to
///    `[-leftPadding[i]]` so each row's first REAL token lands at RoPE position
///    0 regardless of how much it was padded. This is the ``BatchPositionedKVCache``
///    contract A1 introduced, now carrying a genuinely per-row offset (A1's
///    wrapper only replicated one shared scalar).
///  - **Left-padding-aware mask** (`makeMask`) that masks each row's pad columns
///    at EVERY step — prefill AND decode — so pad tokens never contribute to
///    attention. Unlike stock caches, this returns an array mask even at `n == 1`.
///
/// ## Relationship to the Python original
/// A near-mechanical port. The one intentional Swift divergence is that Python
/// mutates `offset`/`keys`/`values` in place (`self.offset += S`) whereas the
/// MLX-Swift ops here are functional (`perRowOffset = perRowOffset + S` rebinds a
/// fresh array); observable state is identical, and it makes `batchOffset`
/// snapshot-safe without copying (a handed-out reference can never be mutated
/// from under a caller). Python's scalar `offset` (the per-row array) maps to
/// ``batchOffset``; Python's `_idx` (the scalar filled length) maps to the
/// `KVCache` protocol's `offset: Int`.
///
/// ## Surface parity (`filter`/`extend`/`extract`)
/// `filter(batchIndices:)`, `extend(other:)`, and `extract(_:)` are the
/// admission/eviction primitives the A2c scheduler needs (row eviction, cohort
/// merge, single-row hand-off). They are ported and parity-tested here but are
/// NOT yet wired into any live decode loop — that seam is A2c.
///
/// ## Isolation
/// Holds non-`Sendable` MLX state; like ``BatchPositionedCacheWrapper`` it is a
/// `final class` used within a single isolation domain (inside
/// `ModelContainer.perform` / the A2c scheduler actor), never shared across
/// tasks.
///
/// ## Deferred to a follow-up wave (NOT ported here)
///  - **`BatchRotatingKVCache`** — the sliding-window sibling
///    (`cache.py:1133`). Its rotation + `dynamic_roll` + rolled-mask math is
///    materially more complex and higher-risk than this dense cache; the master
///    plan explicitly permits deferring sliding-window batching and gating those
///    models out of v1. ``BatchCacheConverter`` enforces that gate (it refuses
///    any `RotatingKVCache`), so Gemma-style hybrid models fall back to the
///    non-batched path until it lands.
///  - **Right-padded chunked prefill** — Python's `prepare(right_padding:)` +
///    `finalize()` roll (`dynamic_roll`) that canonicalises right-padded prefill
///    to left-padded form. v1 does a single left-pad prefill (plan §"Chunked
///    ragged prefill batching"), so `finalize()` here is the protocol no-op.
///  - **Scheduler seam** — wiring these caches into a live ragged decode loop
///    (leftPad → convert → batched prefill → decode) is A2c; A2a's equal-length
///    ``BatchDecodeRunner`` path is untouched. ``BatchKVCacheModelTests`` drives
///    the loop by hand to prove the cache end-to-end without that seam.
public final class BatchKVCache: BatchPositionedKVCache {
    /// Buffer growth granularity along the sequence axis (mirrors Python `step`).
    private let step = 256

    /// Key buffer `[B, H, S, D]`, or `nil` before the first `update`.
    private var keys: MLXArray?
    /// Value buffer `[B, H, S, D]`, or `nil` before the first `update`.
    private var values: MLXArray?

    /// Per-row left-padding amounts, shape `[B]` (Python `left_padding`).
    private var leftPadding: MLXArray
    /// Per-row RoPE offset, shape `[B]` (Python `offset`). Starts at
    /// `[-leftPadding[i]]`; advances by the step width on each `update`.
    private var perRowOffset: MLXArray
    /// Scalar count of filled sequence positions (Python `_idx`). Distinct from
    /// the per-row `perRowOffset`; this is the buffer's logical length.
    private var idx: Int

    /// Build an empty cache for a left-padded cohort.
    ///
    /// - Parameter leftPadding: per-row pad amounts, i.e. `Lmax - realLength[i]`
    ///   for a cohort left-padded to width `Lmax`. See ``BatchPrefillAssembly``
    ///   for computing this from raw prompt lengths.
    public init(leftPadding: [Int]) {
        self.leftPadding = MLXArray(leftPadding.map { Int32($0) })
        self.perRowOffset = MLXArray(leftPadding.map { Int32(-$0) })
        self.idx = 0
    }

    /// Component initialiser used by ``copy()`` and cohort merges.
    private init(
        leftPadding: MLXArray, perRowOffset: MLXArray, idx: Int,
        keys: MLXArray?, values: MLXArray?
    ) {
        self.leftPadding = leftPadding
        self.perRowOffset = perRowOffset
        self.idx = idx
        self.keys = keys
        self.values = values
    }

    // MARK: - BatchPositionedKVCache

    /// Per-row RoPE offsets, shape `[B]`. Read (pre-`update`) by the model as
    /// `.batch(batchOffset + 0)`; the functional `perRowOffset` update keeps any
    /// earlier snapshot stable.
    public var batchOffset: MLXArray { perRowOffset }

    // MARK: - KVCache: core

    /// Scalar filled length (Python `_idx`), NOT the per-row offset.
    public var offset: Int { idx }

    /// No bound — `BatchKVCache` is a full (non-rotating) cache.
    public var maxSize: Int? { nil }

    /// Append `keys`/`values` (`[B, H, S, D]`) and return every cached key/value
    /// up to the new length. Mirrors Python `update_and_fetch`.
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = idx
        let sequence = keys.dim(2)

        let mustGrow: Bool
        if let buffer = self.keys {
            mustGrow = (previous + sequence) > buffer.dim(2)
        } else {
            mustGrow = true
        }
        if mustGrow {
            let batch = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)
            let steps = (step + sequence - 1) / step
            let newKeys = MLXArray.zeros([batch, kvHeads, steps * step, kHeadDim], dtype: keys.dtype)
            let newValues = MLXArray.zeros(
                [batch, kvHeads, steps * step, vHeadDim], dtype: values.dtype)
            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newKeys], axis: 2)
                self.values = concatenated([currentValues, newValues], axis: 2)
            } else {
                self.keys = newKeys
                self.values = newValues
            }
        }

        perRowOffset = perRowOffset + sequence
        idx += sequence
        self.keys?[.ellipsis, previous ..< idx, 0...] = keys
        self.values?[.ellipsis, previous ..< idx, 0...] = values

        guard let bufferKeys = self.keys, let bufferValues = self.values else {
            // Unreachable: the growth branch always assigns both buffers.
            return (keys, values)
        }
        return (bufferKeys[.ellipsis, ..<idx, 0...], bufferValues[.ellipsis, ..<idx, 0...])
    }

    /// Left-padding-aware attention mask for `n` new query rows against the
    /// currently cached keys. Ports Python `BatchKVCache.make_mask` →
    /// `create_causal_mask(N, offset=_idx, left_padding=…)`.
    ///
    /// Independent of upstream `createCausalMask` (which supports right-side
    /// `lengths`, not `leftPadding`): builds the causal core over `[n, idx + n]`
    /// then ANDs a per-row `leftPadding[b] <= keyPosition` term, yielding a
    /// `[B, 1, n, idx + n]` mask. Always an array mask — even at `n == 1`, where
    /// the decode step still must mask each row's pad columns (stock caches
    /// return `.none` there; ragged correctness forbids that shortcut).
    /// Perf note for A2c: the mask depends only on `idx`/`leftPadding`, so the
    /// model loop can compute it once per forward and share it across layers
    /// instead of paying a per-layer allocation every decode step.
    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .array(leftPaddedCausalMask(n: n, offset: idx, windowSize: windowSize))
    }

    private func leftPaddedCausalMask(n: Int, offset: Int, windowSize: Int?) -> MLXArray {
        let rightIndices = MLXArray(Int32(0) ..< Int32(offset + n))
        let leftIndices =
            offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rightIndices
        let queryPositions = leftIndices.expandedDimensions(axis: 1)  // [n, 1]
        let keyPositions = rightIndices.expandedDimensions(axis: 0)  // [1, offset + n]
        var mask = queryPositions .>= keyPositions  // [n, offset + n]
        if let windowSize {
            mask = mask & (queryPositions .< keyPositions + windowSize)
        }
        // leftPadding [B] → [B, 1, 1, 1]; `keyPosition >= leftPadding[b]` masks
        // each row's pad columns. Broadcasts to [B, 1, n, offset + n].
        let padColumns = leftPadding.expandedDimensions(axes: [1, 2, 3])
        return mask & (keyPositions .>= padColumns)
    }

    // MARK: - KVCache: trimming / serialization

    public var isTrimmable: Bool { true }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = min(idx, n)
        idx -= trimmed
        perRowOffset = perRowOffset - trimmed
        return trimmed
    }

    /// `[keys, values, perRowOffset, leftPadding]` — the 4-array form of Python's
    /// `state` tuple (keys/values trimmed to `idx`). Empty before the first
    /// `update`.
    public var state: [MLXArray] {
        get {
            guard let bufferKeys = keys, let bufferValues = values else { return [] }
            if idx < bufferKeys.dim(2) {
                return [
                    bufferKeys[.ellipsis, ..<idx, 0...], bufferValues[.ellipsis, ..<idx, 0...],
                    perRowOffset, leftPadding,
                ]
            }
            return [bufferKeys, bufferValues, perRowOffset, leftPadding]
        }
        set {
            // Lax on purpose (upstream `BaseKVCache` fatalErrors on bad input):
            // this cache is never serialized in v1 — the batched path bypasses
            // the prompt cache — so a malformed payload here can only come from
            // a caller bug, and silently ignoring beats crashing the server.
            guard newValue.count == 4 else { return }
            keys = newValue[0]
            values = newValue[1]
            perRowOffset = newValue[2]
            leftPadding = newValue[3]
            idx = newValue[0].dim(2)
        }
    }

    /// `BatchKVCache` keeps its live metadata (`perRowOffset`, `leftPadding`) in
    /// ``state``, so — like Python, which leaves `meta_state` at the `_BaseCache`
    /// default — there is nothing here. v1 never serialises this cache (the
    /// batched path bypasses the prompt cache).
    public var metaState: [String] {
        get { [""] }
        set {}
    }

    public func copy() -> any KVCache {
        BatchKVCache(
            leftPadding: leftPadding + 0,
            perRowOffset: perRowOffset + 0,
            idx: idx,
            keys: keys.map { $0[.ellipsis] },
            values: values.map { $0[.ellipsis] })
    }

    // MARK: - KVCache: prepare

    /// `prepare(lengths:)` is a no-op for `BatchKVCache`: the per-row left-padding
    /// it needs is fixed at construction, and Python's `prepare` acts only on
    /// `left_padding`/`right_padding` (never `lengths`, which is the rotating
    /// sibling's concern). Right-padded chunked prefill — the only path that adds
    /// padding after construction and needs a `finalize()` roll — is deferred (see
    /// the "Deferred to a follow-up wave" note on the type), so `finalize()` here
    /// is the protocol's default no-op.
    public func prepare(lengths: [Int]?) {}

    public func prepare(lengths: MLXArray?) {}

    // MARK: - Admission / eviction primitives (A2c surface)

    /// Keep only the rows in `batchIndices` (axis-0 gather), then left-shift the
    /// buffer by the minimum surviving left-padding to reclaim now-common pad
    /// columns. Ports Python `filter`. In place.
    public func filter(batchIndices: MLXArray) {
        // Contract: callers must pass at least one surviving index (mirrors
        // Python, where an empty gather would make `.min()` raise). An empty
        // filter is a scheduler bug; leave state unchanged instead of trapping.
        guard batchIndices.size > 0 else { return }
        if let bufferKeys = keys, let bufferValues = values {
            keys = bufferKeys[batchIndices]
            values = bufferValues[batchIndices]
        }
        perRowOffset = perRowOffset[batchIndices]
        leftPadding = leftPadding[batchIndices]

        // Host-sync point: `.item()` forces an eval + device→host copy (same as
        // Python's `.min().item()`). A2c should batch evictions accordingly.
        let minLeftPadding = Int(leftPadding.min().item(Int32.self))
        if minLeftPadding > 0 {
            if let bufferKeys = keys, let bufferValues = values {
                keys = bufferKeys[.ellipsis, minLeftPadding..., 0...]
                values = bufferValues[.ellipsis, minLeftPadding..., 0...]
            }
            idx -= minLeftPadding
            leftPadding = leftPadding - minLeftPadding
        }
    }

    /// Concatenate `other`'s rows onto this cache, right-justifying both to the
    /// larger `idx` so every row stays aligned to the current decode position.
    /// Ports Python `extend`. In place.
    public func extend(other: BatchKVCache) {
        if keys == nil && other.keys == nil {
            leftPadding = concatenated([leftPadding, other.leftPadding], axis: 0)
            perRowOffset = concatenated([perRowOffset, other.perRowOffset], axis: 0)
            return
        }

        let maxIdx = max(idx, other.idx)
        // Head count, key/value head dims, and dtype come from whichever cache
        // has content (both share them when both are non-empty).
        let reference = keys != nil ? self : other
        guard let referenceKeys = reference.keys, let referenceValues = reference.values else {
            return
        }
        let heads = referenceKeys.dim(1)
        let keyDim = referenceKeys.dim(3)
        let valueDim = referenceValues.dim(3)
        let dtype = referenceKeys.dtype
        let maxSizeAxis = max(self.keys?.dim(2) ?? 0, other.keys?.dim(2) ?? 0)

        func padded(_ cache: BatchKVCache) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
            var cacheKeys: MLXArray
            var cacheValues: MLXArray
            if let existingKeys = cache.keys, let existingValues = cache.values {
                cacheKeys = existingKeys
                cacheValues = existingValues
            } else {
                let rows = cache.perRowOffset.dim(0)
                cacheKeys = MLXArray.zeros([rows, heads, 0, keyDim], dtype: dtype)
                cacheValues = MLXArray.zeros([rows, heads, 0, valueDim], dtype: dtype)
            }
            let left = maxIdx - cache.idx
            var right = maxSizeAxis - cacheKeys.dim(2) - left
            if right < 0 {
                cacheKeys = cacheKeys[.ellipsis, ..<right, 0...]
                cacheValues = cacheValues[.ellipsis, ..<right, 0...]
                right = 0
            }
            if left != 0 || right != 0 {
                let widths: [IntOrPair] = [[0, 0], [0, 0], [left, right], [0, 0]]
                cacheKeys = MLX.padded(cacheKeys, widths: widths)
                cacheValues = MLX.padded(cacheValues, widths: widths)
            }
            return (cacheKeys, cacheValues, cache.perRowOffset, cache.leftPadding + left)
        }

        let (selfKeys, selfValues, selfOffset, selfLeftPadding) = padded(self)
        let (otherKeys, otherValues, otherOffset, otherLeftPadding) = padded(other)
        keys = concatenated([selfKeys, otherKeys], axis: 0)
        values = concatenated([selfValues, otherValues], axis: 0)
        perRowOffset = concatenated([selfOffset, otherOffset], axis: 0)
        leftPadding = concatenated([selfLeftPadding, otherLeftPadding], axis: 0)
        idx = maxIdx
    }

    /// Extract row `row` into a standalone `KVCacheSimple`, dropping that row's
    /// left-padding. Ports Python `extract`; the returned cache's `offset` is the
    /// row's real length (`idx - leftPadding[row]`). Uses the public `state`
    /// setter (which sets `offset = keys.dim(2)`) because `KVCacheSimple.keys` is
    /// not settable from outside MLXLMCommon. Extracting before the first
    /// `update` returns an EMPTY cache (Python would raise) — lenient on purpose;
    /// callers own the "cohort was actually prefilled" invariant.
    public func extract(_ row: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        guard let bufferKeys = keys, let bufferValues = values else { return cache }
        // Host-sync point: `.item()` forces an eval + device→host copy.
        let padding = Int(leftPadding[row].item(Int32.self))
        let rowKeys = contiguous(bufferKeys[row ..< (row + 1), 0..., padding ..< idx, 0...])
        let rowValues = contiguous(bufferValues[row ..< (row + 1), 0..., padding ..< idx, 0...])
        cache.state = [rowKeys, rowValues]
        return cache
    }

    // MARK: - Evaluatable

    public func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }
}
