// Copyright © 2026 macMLX. English comments only.

import MLX
import MLXLMCommon

/// Production ``BatchStepEvaluator``: runs a real ``LanguageModel`` forward over
/// a batch-positioned KV cache and samples per row, reusing upstream's public
/// `LogitSampler`/`LogitProcessor` rather than reimplementing sampling.
///
/// ## Sampling
///  - **Greedy fast-path** (all rows `temperature == 0`, no penalty processors):
///    one `argMax(logits, axis: -1)` over the whole `[B, vocab]` — the per-row
///    argmax is free and already batched.
///  - **Per-row loop** (any row needs temperature / top-p / penalties): slice
///    `logits[r]` to `[1, vocab]`, apply that row's processor then sampler, and
///    feed the sampled token back to the row's processor. `B` is small so the
///    loop is cheap. Each row owns its own processor instance because penalty
///    processors hold a single-sequence `TokenRing`.
///
/// ## RoPE correctness / regime
/// `caches` are the ``batchPositioned(_:batch:)`` wrappers, so decode RoPE takes
/// the per-row `.batch` array-offset kernel (A1's fix). A2a is an EQUAL-LENGTH,
/// lockstep cohort: all rows share one advancing offset, which is exactly what
/// A1's single-shared-offset wrapper replicates correctly. Masked (finished)
/// rows keep being fed a pad token so the batch shape and the shared offset
/// advance uniformly — the cohort never desynchronizes, keeping it inside A1's
/// proven regime (ragged / per-row offsets are A2b).
///
/// ## Isolation
/// Holds non-`Sendable` MLX state (`model`, `caches`, samplers, processors).
/// Not `Sendable`: construct and drive within one isolation domain (inside
/// `ModelContainer.perform`, or the A2c scheduler actor).
final class ModelBatchStepEvaluator: BatchStepEvaluator {
    private let model: any LanguageModel
    private let caches: [KVCache]
    private let batch: Int
    private let samplers: [LogitSampler]
    private var processors: [LogitProcessor?]
    private let allGreedy: Bool

    /// - Parameters:
    ///   - model: the resident language model (its forward reads `cache.ropeOffset`).
    ///   - caches: batch-positioned caches from ``batchPositioned(_:batch:)``.
    ///   - cohortParameters: per-row generation parameters (`B` entries); their
    ///     `.sampler()` / `.processor()` factories build the per-row sampling.
    init(model: any LanguageModel, caches: [KVCache], cohortParameters: [GenerateParameters]) {
        self.model = model
        self.caches = caches
        self.batch = cohortParameters.count
        self.samplers = cohortParameters.map { $0.sampler() }
        let processors = cohortParameters.map { $0.processor() }
        self.processors = processors
        // Greedy fast-path is valid only when NO row needs temperature-based
        // sampling or a penalty processor: then every row's token is the plain
        // per-row argmax, computable in one batched op.
        self.allGreedy =
            cohortParameters.allSatisfy { $0.temperature == 0 }
            && processors.allSatisfy { $0 == nil }
    }

    func prefill(_ promptRows: [[Int]]) throws -> [Int] {
        let promptLength = promptRows.first?.count ?? 0
        let flat = promptRows.flatMap { $0 }
        let inputs = MLXArray(flat, [batch, promptLength])

        // Seed each row's penalty processor with its own prompt (no-op on the
        // greedy fast-path, where every processor is nil).
        for row in 0..<batch where processors[row] != nil {
            processors[row]?.prompt(MLXArray(promptRows[row]))
        }

        let logits = model(inputs, cache: caches)
        return sampleLastPosition(logits)
    }

    func step(_ fed: [Int]) throws -> [Int] {
        let inputs = MLXArray(fed, [batch, 1])
        let logits = model(inputs, cache: caches)
        return sampleLastPosition(logits)
    }

    /// Reduce `[B, S, vocab]` logits to the `B` next tokens using the last
    /// sequence position.
    private func sampleLastPosition(_ logits: MLXArray) -> [Int] {
        let sequenceLength = logits.dim(1)
        let last = logits[0..., (sequenceLength - 1)..., 0...].reshaped(batch, -1)  // [B, vocab]

        if allGreedy {
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
