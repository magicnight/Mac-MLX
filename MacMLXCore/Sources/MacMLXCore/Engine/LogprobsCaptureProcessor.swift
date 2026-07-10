// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXNN
import MLXLMCommon

/// A pass-through ``LogitProcessor`` that CAPTURES each step's logprobs without
/// altering them — the decode-time source for OpenAI `logprobs` output. Installed
/// as the OUTERMOST processor so it observes the exact (post-`logit_bias`,
/// post-penalty) distribution the sampler sees, matching mlx-lm's
/// `logprobs = logsoftmax(logits)` computed after the logits processors and
/// before the sampler.
///
/// For each generated token it records the token's own logprob plus the top-N
/// alternatives into a thread-safe ``Sink`` the engine drains one entry per
/// emitted token. `process(logits:)` returns its input unchanged.
public struct LogprobsCaptureProcessor: LogitProcessor {

    /// Thread-safe FIFO of captured entries. The generation (iterator) task
    /// APPENDS during ``didSample(token:)`` while the engine's stream loop POPS
    /// one entry per emitted token from a DIFFERENT task, so every access is
    /// locked — an unsynchronized `Array` append/read across tasks would race and
    /// can reallocate mid-read.
    public final class Sink: @unchecked Sendable {
        public struct Entry: Sendable {
            public let tokenID: Int
            public let logprob: Float
            public let top: [(id: Int, logprob: Float)]
        }
        private let lock = NSLock()
        private var buffer: [Entry] = []
        public init() {}
        func append(_ entry: Entry) {
            lock.lock(); buffer.append(entry); lock.unlock()
        }
        /// Oldest captured entry, or nil when none is buffered yet.
        public func popFirst() -> Entry? {
            lock.lock(); defer { lock.unlock() }
            return buffer.isEmpty ? nil : buffer.removeFirst()
        }
    }

    /// Optional processors (logit_bias, penalties) applied before capture.
    public var inner: LogitProcessor?
    private let sink: Sink
    private let topN: Int

    /// Holds the current step's logprob vector + top-N between the non-mutating
    /// ``process(logits:)`` and ``didSample(token:)``. Both run sequentially in
    /// the single generation task, so no cross-task access — unlike ``Sink``.
    private final class StepBox: @unchecked Sendable {
        var logprobs: MLXArray?
        var topIDs: [Int] = []
        var topVals: [Float] = []
    }
    private let step = StepBox()

    public init(inner: LogitProcessor?, sink: Sink, topN: Int) {
        self.inner = inner
        self.sink = sink
        self.topN = Swift.max(0, topN)
    }

    public mutating func prompt(_ prompt: MLXArray) {
        inner?.prompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        let processed = inner?.process(logits: logits) ?? logits
        let vocab = processed.dim(-1)
        let flat = processed.reshaped([vocab])
        let lp = logSoftmax(flat, axis: -1)
        step.logprobs = lp
        if topN > 0 {
            let k = Swift.min(topN, vocab)
            // argSort of the negated logprobs ⇒ descending-by-logprob ids.
            let descending = argSort((-lp).asType(.float32), axis: -1)
            let topDesc = descending[0 ..< k]
            let topVals = take(lp, topDesc, axis: 0)
            eval(topDesc, topVals)
            step.topIDs = topDesc.asArray(Int32.self).map(Int.init)
            step.topVals = topVals.asArray(Float.self)
        } else {
            step.topIDs = []
            step.topVals = []
        }
        return processed
    }

    public mutating func didSample(token: MLXArray) {
        inner?.didSample(token: token)
        let id = token.item(Int.self)
        let logprob: Float = step.logprobs.map { $0[id].item(Float.self) } ?? 0
        let top = zip(step.topIDs, step.topVals).map { (id: $0.0, logprob: $0.1) }
        sink.append(Sink.Entry(tokenID: id, logprob: logprob, top: top))
    }
}
