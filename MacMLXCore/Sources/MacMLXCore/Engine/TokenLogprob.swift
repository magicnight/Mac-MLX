// Copyright © 2026 macMLX. English comments only.

import Foundation

/// One generated token's logprob and its top-N alternatives (OpenAI
/// `logprobs.content[]`). Emitted per generated token when a request sets
/// `logprobs: true`; the engine computes `logprob = logSoftmax(logits)[token]`
/// on the same (post-processor, pre-sampler) distribution mlx-lm reports.
public struct TokenLogprob: Codable, Hashable, Sendable {
    /// The decoded text of this token (may include a leading-space marker, and
    /// may be a partial UTF-8 fragment for byte-level BPE tokenizers).
    public let token: String
    /// Natural-log probability of `token` under the sampled distribution.
    public let logprob: Float
    /// UTF-8 bytes of `token` (OpenAI `bytes`). Nil when the token text isn't
    /// representable as concrete bytes.
    ///
    /// Best-effort caveat (v1): the bytes come from decoding the single token
    /// id. A byte-level BPE token that is a PARTIAL UTF-8 fragment (common for
    /// CJK/emoji split across tokens) decodes to U+FFFD, so `bytes` will be
    /// the replacement character's bytes ([239, 191, 189]) rather than the
    /// token's true raw bytes — clients cannot reconstruct split characters
    /// from them. Fixing this requires the tokenizer's raw byte mapping
    /// (byte-level pre-tokenizer inverse), which the public Tokenizer API
    /// does not expose today. `token` text and `logprob` are unaffected.
    public let bytes: [Int]?
    /// The `top_logprobs` most-likely alternatives at this position, highest
    /// logprob first. Empty when `top_logprobs` was 0.
    public let topLogprobs: [Alternative]

    public init(token: String, logprob: Float, bytes: [Int]?, topLogprobs: [Alternative]) {
        self.token = token
        self.logprob = logprob
        self.bytes = bytes
        self.topLogprobs = topLogprobs
    }

    /// One alternative token considered at a position (OpenAI
    /// `logprobs.content[].top_logprobs[]`).
    public struct Alternative: Codable, Hashable, Sendable {
        public let token: String
        public let logprob: Float
        public let bytes: [Int]?

        public init(token: String, logprob: Float, bytes: [Int]?) {
            self.token = token
            self.logprob = logprob
            self.bytes = bytes
        }
    }
}
