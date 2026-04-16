/// Lifecycle state of an inference engine.
///
/// Transitions are: `.idle → .loading → .ready → .generating → .ready → .idle`.
/// Any state may transition to `.error(_)`.
public enum EngineStatus: Equatable, Hashable, Sendable {
    case idle
    case loading(model: String)
    case ready(model: String)
    case generating
    case error(String)

    /// True only when a model is loaded and not currently generating.
    public var isLoaded: Bool {
        if case .ready = self { return true }
        return false
    }
}
