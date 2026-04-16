import Foundation

/// Failures at the inference service layer (subprocess management, port allocation, etc.).
public enum InferenceServiceError: LocalizedError, Equatable, Sendable {
    case pythonNotFound
    case portAlreadyInUse(Int)
    case noAvailablePort
    case backendCrashed(exitCode: Int32)
    case modelNotLoaded(String)

    public var errorDescription: String? {
        switch self {
        case .pythonNotFound:
            return "Python 3.10+ not found. Please install via Homebrew or set the path in Settings."
        case .portAlreadyInUse(let port):
            return "Port \(port) is already in use."
        case .noAvailablePort:
            return "No available port found in the searched range."
        case .backendCrashed(let code):
            return "The backend process crashed (exit code \(code))."
        case .modelNotLoaded(let id):
            return "Model is not loaded: \(id)."
        }
    }
}
