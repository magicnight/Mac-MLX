// Copyright © 2026 macMLX. English comments only.

import MLX
import MLXLMCommon

/// Applies an ordered list of ``LogitProcessor``s in sequence — the composition
/// primitive that lets macMLX stack per-request processors (e.g. `logit_bias`
/// first, then the repetition/penalty processor, matching mlx-lm's
/// `make_logits_processors` order) into the single `inner` slot the constrained
/// decoder and the plain custom path both expect. Element 0 runs first.
public struct ChainedLogitProcessor: LogitProcessor {
    private var processors: [any LogitProcessor]

    /// - Returns: nil when the (compacted) list is empty, so the caller installs
    ///   no processor at all rather than an inert wrapper.
    public init?(_ processors: [(any LogitProcessor)?]) {
        let compact = processors.compactMap { $0 }
        guard !compact.isEmpty else { return nil }
        self.processors = compact
    }

    public mutating func prompt(_ prompt: MLXArray) {
        for i in processors.indices { processors[i].prompt(prompt) }
    }

    public func process(logits: MLXArray) -> MLXArray {
        var out = logits
        for processor in processors { out = processor.process(logits: out) }
        return out
    }

    public mutating func didSample(token: MLXArray) {
        for i in processors.indices { processors[i].didSample(token: token) }
    }
}
