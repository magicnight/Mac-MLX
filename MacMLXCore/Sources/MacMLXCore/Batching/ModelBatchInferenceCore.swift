// Copyright © 2026 macMLX. English comments only.

import MLX
import MLXLMCommon

/// Production ``BatchInferenceCore``: runs a real ``LanguageModel`` over a
/// growing/shrinking ragged ``BatchKVCache`` cohort, implementing the A2c
/// continuous-batching primitives on top of the A2b cache surface.
///
/// ## The prefill/decode split (works around mlx-swift#441)
///  - **Admit = B=1 prefill + merge.** Each new request is prefilled on its own
///    through a fresh stock `KVCacheSimple` — a single-row, scalar-offset
///    forward, i.e. the ordinary single-stream path that has no batched-RoPE
///    hazard. Its first token is sampled from the last prompt position, its
///    filled buffers are wrapped as a one-row ``BatchKVCache`` (zero
///    left-padding), and that row is ``BatchKVCache/extend(other:)``ed onto the
///    running batch, which left-pads it to the cohort width. Multi-row prefill
///    is never batched, so the corrupt batched-prefill RoPE (ml-explore/
///    mlx-swift#441) is never exercised.
///  - **Decode = batched L=1.** Every generation step is a single-token
///    `[B, 1]` forward over the running ``BatchKVCache`` list, whose per-row
///    `.batch` RoPE offset is the A1-proven correct kernel path. This is the
///    only phase that batches across rows.
///
/// ## Sampling (reuses upstream, per row)
/// Greedy fast-path (`argMax` over `[B, vocab]`) when every live row is
/// `temperature == 0` with no penalty processor; otherwise the per-row loop
/// (slice `logits[r]`, apply that row's `LogitProcessor`/`LogitSampler`), exactly
/// as ``ModelBatchStepEvaluator``. Each row owns its sampler/processor because
/// penalty processors hold a single-sequence `TokenRing`; those parallel arrays
/// are appended on ``admit(_:)`` and gathered on ``evict(keeping:)`` so they stay
/// row-aligned with the cache through admission and eviction.
///
/// ## Coverage gate
/// On the FIRST ``admit(_:)`` the model's fresh caches are probed with
/// ``BatchCacheConverter`` (dense-cache-type gate) AND
/// ``BatchModelAllowlist`` (verified-`ropeOffset` architecture gate). Failing
/// either throws ``BatchUnsupportedError/cacheNotBatchable`` before any row is
/// admitted, so an uncoverable model never produces a half-built cohort.
///
/// ## Isolation
/// Holds non-`Sendable` MLX state (`model`, running `[BatchKVCache]`, samplers,
/// processors). Not `Sendable`: constructed and driven within one isolation
/// domain (the ``BatchScheduler`` actor / `ModelContainer.perform`).
final class ModelBatchInferenceCore: BatchInferenceCore {
    private let model: any LanguageModel

    /// Running batch, one ``BatchKVCache`` per KV-owning layer, or `nil` before
    /// the first admit / after the batch fully drains.
    private var batchCaches: [BatchKVCache]?

    /// Per-row samplers/processors, row-aligned with the running batch.
    private var samplers: [LogitSampler] = []
    private var processors: [LogitProcessor?] = []
    /// Per-row greedy flag (`temperature == 0` and no processor); its AND is the
    /// batched `argMax` fast-path gate.
    private var greedyRows: [Bool] = []

    /// `false` until the first coverage probe PASSES. A failed probe is NOT
    /// cached: every retry re-probes and re-throws, so an uncovered model can
    /// never slip past the gate on a later submit (the probe is cheap — type
    /// checks plus an empty `newCache` array, no MLX compute).
    private var coveragePassed = false

    init(model: any LanguageModel) {
        self.model = model
    }

    var rowCount: Int { samplers.count }

    // MARK: - Admit (B=1 prefill + merge)

    func admit(_ rows: [BatchSlotConfig]) throws -> [Int] {
        guard !rows.isEmpty else { return [] }
        try ensureCoverage()

        var firstTokens: [Int] = []
        firstTokens.reserveCapacity(rows.count)
        for config in rows {
            firstTokens.append(try admitOne(config))
        }
        return firstTokens
    }

    /// Prefill one request B=1, merge its row onto the running batch, and return
    /// its first sampled token.
    private func admitOne(_ config: BatchSlotConfig) throws -> Int {
        // Fresh stock caches for this row's isolated B=1 prefill. Dense by the
        // coverage gate, so wrapping is unnecessary — a single row at a scalar
        // offset is the correct (non-batched) RoPE path.
        let stock = model.newCache(parameters: nil)
        let promptLength = config.promptTokens.count
        let promptArray = MLXArray(config.promptTokens.map { Int32($0) }, [1, promptLength])

        let sampler = config.parameters.sampler()
        var processor = config.parameters.processor()
        // Seed the row's penalty processor with its own prompt (no-op when nil).
        // All sampling mutations must land on the SAME instance appended below,
        // hence the local `var` rather than a helper that copies it by value.
        processor?.prompt(promptArray)

        let logits = model(promptArray, cache: stock)
        let sequenceLength = logits.dim(1)
        var last = logits[0..., (sequenceLength - 1)..., 0...].reshaped(1, -1)  // [1, vocab]
        last = processor?.process(logits: last) ?? last
        let sampled = sampler.sample(logits: last)  // [1]
        sampled.eval()
        processor?.didSample(token: sampled)
        let token = sampled.item(Int.self)

        // Wrap each filled stock layer as a one-row BatchKVCache and merge. The
        // caches are `final class`es, so mutating an element mutates the shared
        // instance `batchCaches` already references — no array write-back needed.
        let newRow = stock.map { layer -> BatchKVCache in
            let state = layer.state
            return BatchKVCache.singleRow(keys: state[0], values: state[1])
        }
        if let caches = batchCaches {
            for index in caches.indices {
                caches[index].extend(other: newRow[index])
            }
        } else {
            batchCaches = newRow
        }

        samplers.append(sampler)
        processors.append(processor)
        greedyRows.append(config.parameters.temperature == 0 && processor == nil)
        return token
    }

    // MARK: - Decode (batched L=1)

    func decode(_ feedback: [Int]) throws -> [Int] {
        let batch = samplers.count
        guard batch > 0, let caches = batchCaches else { return [] }
        // Defensive contract check on the INPUT side (the scheduler already
        // guards the output count): a desynced feedback array would otherwise
        // surface as an MLX reshape crash instead of a typed error.
        guard feedback.count == batch else {
            throw BatchUnsupportedError.evaluatorContractViolation(
                expected: batch, got: feedback.count)
        }
        let inputs = MLXArray(feedback.map { Int32($0) }, [batch, 1])
        let logits = model(inputs, cache: caches)
        return sampleLastPosition(logits, batch: batch)
    }

    // MARK: - Evict (filter)

    func evict(keeping keepRows: [Int]) {
        let batch = samplers.count
        guard batch > 0 else { return }
        if keepRows.count == batch {
            // Still a real reordering hazard if indices are permuted, but the
            // scheduler only ever passes ascending survivor indices; a full-keep
            // set is therefore identity and needs no cache surgery.
            return
        }
        if keepRows.isEmpty {
            batchCaches = nil
            samplers = []
            processors = []
            greedyRows = []
            return
        }
        let keepArray = MLXArray(keepRows.map { Int32($0) })
        if let caches = batchCaches {
            for index in caches.indices {
                caches[index].filter(batchIndices: keepArray)
            }
        }
        samplers = keepRows.map { samplers[$0] }
        processors = keepRows.map { processors[$0] }
        greedyRows = keepRows.map { greedyRows[$0] }
    }

    // MARK: - Coverage gate

    private func ensureCoverage() throws {
        guard !coveragePassed else { return }
        // Both gates: cache SHAPE (dense-only) and model ARCHITECTURE
        // (verified `ropeOffset`). Either failing means ragged batching would
        // silently corrupt, so refuse before admitting any row. The flag is
        // set only AFTER both gates pass — a failed probe must re-probe (and
        // re-throw) on every subsequent admit, otherwise a second submit
        // against an uncovered model would bypass the gate silently.
        guard BatchModelAllowlist.contains(model),
            BatchCacheConverter.makeBatchCaches(from: model.newCache(parameters: nil), leftPadding: [0])
                != nil
        else {
            throw BatchUnsupportedError.cacheNotBatchable
        }
        coveragePassed = true
    }

    // MARK: - Sampling

    /// Reduce `[B, 1, vocab]` decode logits to the `B` next tokens, greedy
    /// fast-path or per-row loop (mirrors ``ModelBatchStepEvaluator``).
    private func sampleLastPosition(_ logits: MLXArray, batch: Int) -> [Int] {
        let sequenceLength = logits.dim(1)
        let last = logits[0..., (sequenceLength - 1)..., 0...].reshaped(batch, -1)  // [B, vocab]

        if greedyRows.allSatisfy({ $0 }) {
            let tokens = argMax(last, axis: -1)  // [B]
            tokens.eval()
            return tokens.asArray(Int.self)
        }

        var out = [Int]()
        out.reserveCapacity(batch)
        for row in 0..<batch {
            var rowLogits = last[row..<(row + 1)]  // [1, vocab]
            rowLogits = processors[row]?.process(logits: rowLogits) ?? rowLogits
            let sampled = samplers[row].sample(logits: rowLogits)  // [1]
            sampled.eval()
            out.append(sampled.item(Int.self))
            processors[row]?.didSample(token: sampled)
        }
        return out
    }
}
