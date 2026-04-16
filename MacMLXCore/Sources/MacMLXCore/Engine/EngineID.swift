/// Identifies which inference engine implementation handles a request.
public enum EngineID: String, Codable, Hashable, Sendable, CaseIterable {
    /// Apple's mlx-swift-lm package, in-process. Default engine.
    case mlxSwift  = "mlx-swift-lm"
    /// SwiftLM external binary, subprocess. For 100B+ MoE models.
    case swiftLM   = "swift-lm"
    /// Python mlx-lm via uv-managed subprocess. Maximum model compatibility.
    case pythonMLX = "python-mlx"
}
