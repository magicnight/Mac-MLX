// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon

/// The decode-time enforcement point for structured output (Track C): an
/// upstream `LogitProcessor` that masks, at every step, every token whose bytes
/// the active ``ConstraintState`` would reject — so the sampled token can only
/// ever keep the output on a path to a complete, well-formed JSON (C1) or
/// schema-conforming (C2) document.
///
/// ## Composition
/// Any penalty processor the request already carries (`repetitionPenalty`, …)
/// is applied FIRST via ``inner``; the constraint mask is applied LAST, so a
/// penalty can never resurrect a token the grammar forbids.
///
/// ## Masking strategy (correctness-first, with a greedy fast path)
/// Classifying a 150 K-token vocabulary against the grammar every step is the
/// intrinsic cost of constrained decoding — the automaton is sequential CPU
/// logic, so the candidate tokens must be brought to the CPU.
///
///  - **Greedy** (`temperature == 0`, the usual structured-output mode): only
///    the single highest-logit legal token is needed. Tokens are scanned in
///    descending-logit order and the first legal one wins, so typically only a
///    few hundred (`topKCap`) classifications happen per step. A full-vocabulary
///    descending scan is the rare fallback used only when none of the top-K is
///    legal.
///  - **Sampling** (`temperature > 0`): correctness requires masking the *whole*
///    illegal set so the renormalized distribution only contains legal tokens,
///    which is an O(vocabulary) classification each step. This is slower and is
///    reported in the throughput measurement; greedy is recommended for
///    structured output.
///
/// ## State & value semantics
/// The grammar state advances only on the token actually sampled
/// (``didSample(token:)``). The per-model ``TokenVocabularyTable`` is resolved
/// lazily on the first ``process(logits:)`` (the only place the authoritative
/// vocabulary size — the logit dimension — is known) and shared through a
/// reference box so the non-mutating `process` can populate it; it is cached
/// across requests by ``TokenVocabularyCache``.
public struct JSONConstraintProcessor: LogitProcessor {

    /// Reference holder for the lazily-resolved table (and the reusable arange
    /// index used to build the greedy one-hot mask), so the non-mutating
    /// ``process(logits:)`` can memoize them without the struct being mutable
    /// there. Accessed only from the single generation task — no concurrency.
    final class TableBox: @unchecked Sendable {
        var table: TokenVocabularyTable?
        var arange: MLXArray?
    }

    /// Penalty (or other) processor applied before the constraint mask.
    public var inner: LogitProcessor?
    /// The active grammar/schema state; advanced by ``didSample(token:)``.
    public var state: ConstraintState

    private let cache: TokenVocabularyCache
    private let modelID: String
    private let tokenizer: any Tokenizer
    private let stopTokenIDs: Set<Int>
    private let greedy: Bool
    private let topKCap: Int
    private let box = TableBox()

    /// - Parameters:
    ///   - format: the validated response format (C1 or C2).
    ///   - inner: an optional penalty processor to run before the mask.
    ///   - cache: process-wide vocabulary-table cache.
    ///   - modelID: resident model id (cache key).
    ///   - tokenizer: the model's tokenizer, used to build the table on a miss.
    ///   - stopTokenIDs: the model's complete stop/EOS id set — masked until the
    ///     constraint reaches an accepting state.
    ///   - greedy: whether sampling is greedy (`temperature == 0`), enabling the
    ///     top-K fast path.
    ///   - maxDepth: JSON nesting cap for the C1 automaton.
    ///   - topKCap: how many top-logit tokens the greedy path classifies before
    ///     falling back to a full scan.
    public init(
        format: ResponseFormat,
        inner: LogitProcessor?,
        cache: TokenVocabularyCache,
        modelID: String,
        tokenizer: any Tokenizer,
        stopTokenIDs: Set<Int>,
        greedy: Bool,
        maxDepth: Int = 64,
        topKCap: Int = 256
    ) {
        self.state = ConstraintState.initial(for: format, maxDepth: maxDepth)
        self.inner = inner
        self.cache = cache
        self.modelID = modelID
        self.tokenizer = tokenizer
        self.stopTokenIDs = stopTokenIDs
        self.greedy = greedy
        self.topKCap = Swift.max(1, topKCap)
    }

    // MARK: - LogitProcessor

    public mutating func prompt(_ prompt: MLXArray) {
        // The constraint governs only generated tokens; the prompt does not
        // advance the grammar. Still forward it to any inner penalty processor.
        inner?.prompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        let processed = inner?.process(logits: logits) ?? logits
        let vocab = processed.dim(-1)
        let table = resolveTable(vocabularySize: vocab)

        if greedy {
            guard let best = bestLegalIndex(in: processed, table: table, vocab: vocab) else {
                // No legal continuation exists (and EOS is itself illegal because
                // the document is not yet complete). Force a clean end-of-stream
                // instead of emitting unmasked logits — see ``forceTermination``.
                return forceTermination(to: processed, vocab: vocab)
            }
            return applyMask(keepingOnly: best, to: processed, vocab: vocab)
        }

        return applyFullMask(to: processed, table: table, vocab: vocab)
    }

    public mutating func didSample(token: MLXArray) {
        inner?.didSample(token: token)
        guard let table = box.table else { return }
        let id = token.item(Int.self)
        switch table.classification(of: id) {
        case .eos, .unusable:
            // EOS terminates generation; an unusable token should never have
            // been sampled (it is masked). Either way the grammar does not
            // advance.
            return
        case .bytes(let bytes):
            if let next = state.walk(bytes) {
                state = next
            }
            // If the walk fails the token was illegal yet somehow sampled — keep
            // the last valid state rather than corrupting it.
        }
    }

    // MARK: - Masking

    /// Find the single best legal token for the greedy path — the highest-logit
    /// token the constraint accepts, or `nil` when none is legal.
    ///
    /// The MLX here is only the *ranking* (argmax / argsort over the logit
    /// vector); the *decision* it feeds — "first legal token in descending-logit
    /// order" — is the pure, MLX-free ``selectLegalToken(state:table:descendingLogitOrder:)``
    /// / ``isLegal(_:state:table:)`` (unit-tested without Metal).
    private func bestLegalIndex(
        in logits: MLXArray,
        table: TokenVocabularyTable,
        vocab: Int
    ) -> Int? {
        let flat = logits.reshaped([vocab])

        // Fast path: the model's own top token is usually already a legal JSON
        // continuation, so try the global argmax first — one O(vocab) reduction
        // and one classification, no sort and no dtype copy. This is what keeps
        // the greedy overhead small on the common step (the sort below is
        // skipped entirely whenever the model's preferred token is legal).
        let argmax = argMax(flat, axis: -1)
        argmax.eval()
        let top1 = argmax.item(Int.self)
        if Self.isLegal(top1, state: state, table: table) { return top1 }

        // The top token was illegal (the model wanted non-JSON, or the grammar
        // forbids it here). Sort once and scan in descending-logit order: the
        // top-K first, then the remainder. Cast to float32 here (only on this
        // cold path) for a stable sort.
        let ascending = argSort(flat.asType(.float32), axis: -1).asType(.int32)
        let k = Swift.min(topKCap, vocab)
        let topSlice = ascending[(vocab - k) ..< vocab]
        topSlice.eval()
        let top = topSlice.asArray(Int32.self)
        // Descending-logit order of the top-K (`ascending` is ascending), fed to
        // the pure selector so production and the MLX-free tests share one rule.
        let topDescending = (0 ..< k).map { Int(top[k - 1 - $0]) }
        if let id = Self.selectLegalToken(state: state, table: table, descendingLogitOrder: topDescending) {
            return id
        }

        // Rare fallback: none of the top-K was legal — scan the remainder in
        // place (avoid materializing a vocabulary-sized `[Int]` copy).
        ascending.eval()
        let all = ascending.asArray(Int32.self)
        var i = vocab - k - 1
        while i >= 0 {
            let id = Int(all[i])
            if Self.isLegal(id, state: state, table: table) { return id }
            i -= 1
        }
        return nil
    }

    /// Keep exactly one token and forbid all others, built entirely on the GPU
    /// (no vocabulary-sized host allocation or host→device copy): `-inf`
    /// everywhere the reused arange index is not `best`.
    private func applyMask(keepingOnly best: Int, to logits: MLXArray, vocab: Int) -> MLXArray {
        let flat = logits.reshaped([vocab])
        let keep = cachedArange(vocab) .== Int32(best)
        let negInf = MLXArray(-Float.infinity).asType(flat.dtype)
        return MLX.where(keep, flat, negInf).reshaped([1, vocab])
    }

    /// The `[0, vocab)` index vector, built once per model and reused every step
    /// (it is constant for a fixed vocabulary).
    private func cachedArange(_ vocab: Int) -> MLXArray {
        if let arange = box.arange { return arange }
        let arange = MLXArray.arange(vocab)
        box.arange = arange
        return arange
    }

    /// Classify the whole vocabulary and mask every illegal token (sampling
    /// path — correct for any sampler).
    private func applyFullMask(
        to logits: MLXArray,
        table: TokenVocabularyTable,
        vocab: Int
    ) -> MLXArray {
        var mask = [Float](repeating: 0, count: vocab)
        var anyLegal = false
        for id in 0..<vocab {
            if Self.isLegal(id, state: state, table: table) {
                anyLegal = true
            } else {
                mask[id] = -.infinity
            }
        }
        // No legal continuation: force a clean end-of-stream rather than returning
        // an all -inf distribution (which would NaN the sampler) or the unmasked
        // logits (which would let the model spew arbitrary tokens). Same behavior
        // as the greedy path — see ``forceTermination``.
        guard anyLegal else { return forceTermination(to: logits, vocab: vocab) }
        let maskArray = MLXArray(mask).asType(logits.dtype).reshaped([1, vocab])
        return logits + maskArray
    }

    // MARK: - Termination on an all-illegal state

    /// No token is a legal continuation from the current constraint state, and
    /// EOS is itself illegal because the JSON document is not yet complete.
    ///
    /// Emitting the unmasked logits here would let the sampler pick an arbitrary
    /// token and silently break the "always valid JSON" guarantee: the grammar
    /// walk would then fail, the state would freeze, and every subsequent step
    /// would compound garbage into a 200 response that looks like "valid prefix +
    /// junk" with no error. Instead force a clean end-of-stream — log one
    /// diagnostic and mask everything except the lowest stop/EOS id, so the
    /// sampler MUST emit EOS and generation stops on an observably incomplete
    /// JSON prefix the client can detect (truncated body + `finish_reason`).
    /// Greedy and sampling paths share this so their behavior is identical.
    ///
    /// If the model declares no stop token in range there is nothing to force, so
    /// fall back to the unmasked logits (still logged) rather than masking to an
    /// all -inf distribution.
    private func forceTermination(to logits: MLXArray, vocab: Int) -> MLXArray {
        LogManager.shared.logSync(
            "JSONConstraintProcessor: no legal token at automaton state "
                + "[\(state.diagnosticDescription)] — forcing EOS to terminate "
                + "generation (output is an incomplete JSON prefix).",
            level: .error,
            category: .error
        )
        guard let eos = Self.lowestStopToken(in: stopTokenIDs, vocab: vocab) else {
            return logits
        }
        return applyMask(keepingOnly: eos, to: logits, vocab: vocab)
    }

    // MARK: - Pure decision core (MLX-free, unit-tested without Metal)

    /// The greedy decision: given candidate token ids already ranked in
    /// descending-logit order, return the highest-logit token the constraint
    /// accepts, or `nil` when every candidate is illegal (the all-illegal case
    /// that forces EOS). Pure — no MLX — so the masking policy is unit-testable
    /// with a scripted vocabulary and plain logits.
    static func selectLegalToken(
        state: ConstraintState,
        table: TokenVocabularyTable,
        descendingLogitOrder ids: [Int]
    ) -> Int? {
        ids.first { isLegal($0, state: state, table: table) }
    }

    /// Whether token `id` is a legal next token under `state`: a stop/EOS token
    /// only in an accepting (complete) state; an unusable token never; a byte
    /// token iff the automaton accepts its exact bytes. Pure and MLX-free.
    static func isLegal(_ id: Int, state: ConstraintState, table: TokenVocabularyTable) -> Bool {
        switch table.classification(of: id) {
        case .eos:
            return state.isComplete
        case .unusable:
            return false
        case .bytes(let bytes):
            return state.accepts(bytes)
        }
    }

    /// The lowest in-range stop/EOS id to force when generation is wedged, or
    /// `nil` when the model declares none within `0..<vocab`. Deterministic
    /// (the minimum) so the forced-termination token is stable. Pure and
    /// MLX-free.
    static func lowestStopToken(in stopTokenIDs: Set<Int>, vocab: Int) -> Int? {
        stopTokenIDs.filter { $0 >= 0 && $0 < vocab }.min()
    }

    private func resolveTable(vocabularySize: Int) -> TokenVocabularyTable {
        if let table = box.table { return table }
        let table = cache.table(
            modelID: modelID,
            vocabularySize: vocabularySize,
            stopTokenIDs: stopTokenIDs,
            decode: { [tokenizer] id in
                tokenizer.decode(tokenIds: [id], skipSpecialTokens: false)
            }
        )
        box.table = table
        return table
    }
}
