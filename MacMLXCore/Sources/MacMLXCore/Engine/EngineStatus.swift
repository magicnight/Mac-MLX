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

    /// True when a model is loaded — whether it's idle (`.ready`) or in
    /// the middle of producing tokens (`.generating`). Pre-v0.3.1 this
    /// only returned `true` for `.ready`, so the "No model loaded"
    /// banner and input disabled-state flickered on for every send →
    /// first-token window. From the UI's perspective a model that's
    /// generating *is* loaded.
    public var isLoaded: Bool {
        switch self {
        case .ready, .generating:
            return true
        case .idle, .loading, .error:
            return false
        }
    }
}
