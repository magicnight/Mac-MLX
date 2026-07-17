// Copyright © 2026 macMLX. English comments only.

/// The in-process generation configuration the bottleneck classifier can see —
/// the knobs an external monitor cannot, because they live inside the engine
/// that is producing the tokens.
///
/// This is deliberately a small, pure value (no MLX types): the engine fills it
/// in at each generation, and the classifier reads it only to make its ADVICE
/// concrete and actionable. For example, if decode is bandwidth-bound and
/// `kvBits == nil`, the honest suggestion is "try a lower-bit KV cache"; if the
/// KV cache is already 4-bit, that advice would be wrong, so the classifier
/// tailors it. None of these fields feed the bottleneck DECISION itself — that
/// rests on hardware samples plus the phase — they only shape the recommendation.
public struct EngineGenerationConfig: Sendable, Equatable {

    /// KV-cache quantization bit width (mlx-lm `kv_bits`), or `nil` when the KV
    /// cache is full-precision. A non-nil value means quantization is already in
    /// play, so "quantize the KV cache" is no longer a useful suggestion.
    public let kvBits: Int?

    /// KV-cache quantization group size, meaningful only alongside `kvBits`.
    public let kvGroupSize: Int?

    /// Token offset at which KV-cache quantization begins, meaningful only
    /// alongside `kvBits`.
    public let quantizedKVStart: Int?

    /// Number of sequences decoded together this generation. `1` for the
    /// single-stream chat/CLI path; larger under the batch-serving seam. A large
    /// batch shifts decode from bandwidth-bound toward compute-bound (weights are
    /// read once and amortized across the batch), which the advice reflects.
    public let batchSize: Int

    /// True when the KV cache is quantized (`kvBits != nil`).
    public var usesQuantizedKVCache: Bool { kvBits != nil }

    public init(
        kvBits: Int?,
        kvGroupSize: Int?,
        quantizedKVStart: Int?,
        batchSize: Int
    ) {
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.batchSize = batchSize
    }
}
