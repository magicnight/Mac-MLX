// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon

/// One request's contribution to a batched-decode cohort: what to decode and how
/// to sample/stop it. Passed to ``BatchDecodeRunner/make(model:tokenizer:eosTokenIds:cohort:globalMaxTokens:)``,
/// which turns each into a ``BatchDecodeSlot`` + its output stream.
///
/// `Sendable`: it carries only value types, so a caller (the A2c scheduler) can
/// assemble a cohort off-actor and hand it in.
struct BatchSlotConfig: Sendable {
    /// This request's prompt token IDs. Every config in a cohort MUST have the
    /// same length (A2a equal-length regime); the runner rejects ragged cohorts.
    let promptTokens: [Int]

    /// Sampling parameters. `maxTokens` here bounds this row's emitted tokens.
    let parameters: GenerateParameters

    /// Stop strings for this row (incremental, detokenized-stream matching).
    let stopStrings: Set<String>

    init(promptTokens: [Int], parameters: GenerateParameters, stopStrings: Set<String> = []) {
        self.promptTokens = promptTokens
        self.parameters = parameters
        self.stopStrings = stopStrings
    }
}
