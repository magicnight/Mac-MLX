import Foundation

/// Failures originating inside an `InferenceEngine` implementation.
public enum EngineError: LocalizedError, Equatable, Sendable {
    case modelNotLoaded
    case modelNotFound(String)
    case engineNotReady
    case generationInProgress
    case modelLoadFailed(reason: String)
    case unsupportedOperation(String)

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
        case .unsupportedOperation(let op):
            return "Operation not supported by this engine: \(op)."
        }
    }
}
