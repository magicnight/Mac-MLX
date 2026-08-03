import Foundation

/// Failures originating inside an `InferenceEngine` implementation.
public enum EngineError: LocalizedError, Equatable, Sendable {
    case modelNotLoaded
    case modelNotFound(String)
    case engineNotReady
    case generationInProgress
    case modelLoadFailed(reason: String)
    case adapterApplyFailed(reason: String)
    case unsupportedOperation(String)
    /// The draft model requested for speculative decoding (D1) doesn't share
    /// a tokenizer with the target model. mlx-swift-lm's
    /// `SpeculativeTokenIterator` assumes this and does not itself verify
    /// it — a silent mismatch would misinterpret draft-proposed token ids
    /// against the target vocabulary, so macMLX checks explicitly and
    /// throws rather than risking garbled output.
    case draftModelTokenizerMismatch(reason: String)
    /// The requested draft model id (D1, client-controlled wire field —
    /// `GenerateRequest.draftModelID`) isn't safe to use as a filesystem
    /// path component: it contains a path separator, NUL, `..`, a leading
    /// dot, or its resolved directory would fall outside the models root.
    /// See `MLXSwiftEngine.draftModelDirectory(id:)`. The message echoes
    /// only the offending id — never a resolved/standardized path — so it
    /// can't be used by a probing client as a path oracle.
    case invalidDraftModelID(id: String)
    /// An audio operation failed after the model was resident (v0.9 W1a) —
    /// undecodable upload, or a synthesis/transcription forward pass that
    /// threw. Distinct from `modelLoadFailed` so `/v1/audio/*` can tell a bad
    /// request apart from a bad model, and from `generationInProgress` because
    /// nothing about it is a concurrency condition.
    case audioProcessingFailed(reason: String)
    /// The `model` a `/v1/audio/*` request named was rejected locally (v0.9
    /// W1a), before any network call: it is not shaped like a Hugging Face
    /// repo id. Deliberately distinct from `modelLoadFailed` so the handlers
    /// answer 400 rather than 500 — no retry can ever make a malformed id
    /// resolve, and 5xx is exactly what client SDKs retry on by default.
    /// The payload is a complete sentence naming the offending id and the
    /// expected `owner/name` shape (see `AudioEngine.repoIDHint(_:kind:)`), so
    /// `errorDescription` returns it verbatim instead of prefixing it again.
    case invalidAudioModelID(reason: String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "No model is currently loaded."
        case .modelNotFound(let id):
            return "Model not found: \(id)."
        case .engineNotReady:
            return "Engine is not ready. Wait for the current operation to finish."
        case .generationInProgress:
            return "A generation is already in progress on this engine."
        case .modelLoadFailed(let reason):
            return "Model failed to load: \(reason)"
        case .adapterApplyFailed(let reason):
            return "LoRA adapter failed to apply: \(reason)"
        case .unsupportedOperation(let op):
            return "Operation not supported by this engine: \(op)."
        case .draftModelTokenizerMismatch(let reason):
            return "Draft model is incompatible with the target model: \(reason)"
        case .invalidDraftModelID(let id):
            return "Draft model id is not valid: '\(id)'. Ids must be plain names using only "
                + "letters, digits, '.', '_', '-', and must not start with '.' or contain a "
                + "path separator."
        case .audioProcessingFailed(let reason):
            return "Audio processing failed: \(reason)"
        case .invalidAudioModelID(let reason):
            return reason
        }
    }
}
