// Copyright © 2026 macMLX. English comments only.

/// Per-slot incremental detokenizer seam for batched decode.
///
/// Turns one freshly sampled token ID into the incremental text it contributes
/// to that slot's stream (empty string when the token does not yet complete a
/// Unicode scalar). Each ``BatchDecodeSlot`` owns one, mirroring how the single
/// stream path keeps one `NaiveStreamingDetokenizer` per generation.
///
/// This seam exists so the batched-decode SCHEDULING logic (stop strings, EOS,
/// max-tokens, finished-row masking, fan-out) can be exercised in CI with a
/// scripted decoder — no tokenizer, no model, no Metal runtime. The production
/// implementation is ``NaiveIncrementalTextDecoder``.
protocol IncrementalTextDecoder {
    /// Append `token` and return the newly decodable text for this slot.
    mutating func decode(_ token: Int) -> String
}
